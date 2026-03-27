"""
Fused Triton kernels for virtual expert metadata preparation.

These replace multiple small PyTorch ops in the virtual expert LoRA path
with single fused kernels to reduce CUDA kernel launch overhead.
"""

import functools
from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_virtual_topk_ids_kernel(
    topk_ids_ptr,
    token_lora_mapping_ptr,
    virtual_topk_ids_ptr,
    token_lora_mask_ptr,
    num_experts_for_weight: tl.constexpr,
    M,
    top_k: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fuses _get_virtual_topk_ids: comparison + clamp + arithmetic into one kernel.

    For each (m, k):
        lora_id = token_lora_mapping[m]
        mask[m] = (lora_id >= 0)
        safe_lora = max(lora_id, 0)
        if shared_outer:  (handled by num_experts_for_weight == 0 sentinel)
            virtual_topk_ids[m, k] = safe_lora * 1  (= safe_lora)
        else:
            virtual_topk_ids[m, k] = topk_ids[m, k] + safe_lora * num_experts_for_weight
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = M * top_k
    valid = offs < total

    m = offs // top_k
    # k = offs % top_k  # not needed directly

    lora_id = tl.load(token_lora_mapping_ptr + m, mask=valid, other=0)
    mask_val = lora_id >= 0
    safe_lora = tl.maximum(lora_id, 0)

    base = tl.load(topk_ids_ptr + offs, mask=valid, other=0)
    result = base + safe_lora * num_experts_for_weight
    tl.store(virtual_topk_ids_ptr + offs, result, mask=valid)

    # Write mask once per row (at first k position)
    k = offs % top_k
    is_first_k = k == 0
    tl.store(token_lora_mask_ptr + m, mask_val, mask=valid & is_first_k)


def _fused_virtual_topk_ids(
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    num_experts: int,
    shared_outer: bool,
    max_loras: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Fused replacement for _get_virtual_topk_ids().

    Returns: (virtual_topk_ids, token_lora_mask, virtual_num_experts)
    """
    M, top_k = topk_ids.shape
    device = topk_ids.device

    if shared_outer:
        num_experts_for_weight = 1
        # For shared_outer, we need topk_ids to be zeros
        # We create a zeros tensor and pass it as topk_ids to the kernel
        zero_topk = torch.zeros_like(topk_ids)
        input_topk = zero_topk
    else:
        num_experts_for_weight = num_experts
        input_topk = topk_ids

    virtual_topk_ids = torch.empty_like(topk_ids)
    token_lora_mask = torch.empty(M, dtype=torch.bool, device=device)

    BLOCK_SIZE = 1024
    grid = ((M * top_k + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _fused_virtual_topk_ids_kernel[grid](
        input_topk,
        token_lora_mapping,
        virtual_topk_ids,
        token_lora_mask,
        num_experts_for_weight,
        M,
        top_k,
        BLOCK_SIZE,
    )

    virtual_num_experts = num_experts_for_weight * max_loras
    return virtual_topk_ids, token_lora_mask, virtual_num_experts


@triton.jit
def _fused_sanitize_expert_ids_kernel(
    expert_ids_ptr,
    output_ptr,
    num_virtual_experts,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fuses _sanitize_expert_ids: clone + masked fill into one kernel.

    output[i] = expert_ids[i] if expert_ids[i] < num_virtual_experts else -1
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offs < N

    eid = tl.load(expert_ids_ptr + offs, mask=valid, other=0)
    result = tl.where(eid < num_virtual_experts, eid, -1)
    tl.store(output_ptr + offs, result, mask=valid)


def fused_sanitize_expert_ids(
    expert_ids: torch.Tensor,
    num_virtual_experts: int,
) -> torch.Tensor:
    """
    Fused replacement for _sanitize_expert_ids().

    Returns a new tensor with expert_ids >= num_virtual_experts replaced by -1.
    """
    N = expert_ids.numel()
    output = torch.empty_like(expert_ids)

    BLOCK_SIZE = 1024
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _fused_sanitize_expert_ids_kernel[grid](
        expert_ids,
        output,
        num_virtual_experts,
        N,
        BLOCK_SIZE,
    )
    return output


@triton.jit
def _fused_masked_add_kernel(
    output_ptr,
    slice_ptr,
    mask_ptr,
    M,
    top_k: tl.constexpr,
    slice_dim,
    out_stride_m,
    out_stride_k,
    slice_stride_m,
    slice_stride_k,
    slice_offset,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fuses: output[..., offset:offset+dim] += slice * mask[:, None, None]

    For each (m, k, d):
        if mask[m]:
            output[m, k, offset + d] += slice[m, k, d]
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = M * top_k * slice_dim
    valid = offs < total

    d = offs % slice_dim
    mk = offs // slice_dim
    m = mk // top_k
    k = mk % top_k

    mask_val = tl.load(mask_ptr + m, mask=valid, other=False)

    # Only load and store if mask is true
    should_update = valid & mask_val

    slice_val = tl.load(
        slice_ptr + m * slice_stride_m + k * slice_stride_k + d,
        mask=should_update,
        other=0.0,
    )

    out_idx = m * out_stride_m + k * out_stride_k + slice_offset + d
    current = tl.load(output_ptr + out_idx, mask=should_update, other=0.0)
    new_val = current + slice_val
    tl.store(output_ptr + out_idx, new_val, mask=should_update)


def _fused_masked_add(
    output: torch.Tensor,
    output_slice: torch.Tensor,
    token_lora_mask: torch.Tensor,
    slice_offset: int,
) -> None:
    """
    Fused replacement for:
        output[..., offset:offset+dim] += output_slice * mask[:, None, None]

    Modifies output in-place.

    Args:
        output: [M, top_k, total_dim] output tensor
        output_slice: [M, top_k, slice_dim] slice to add
        token_lora_mask: [M] bool mask
        slice_offset: offset into last dim of output
    """
    M, top_k, slice_dim = output_slice.shape
    total = M * top_k * slice_dim

    BLOCK_SIZE = 1024
    grid = ((total + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _fused_masked_add_kernel[grid](
        output,
        output_slice,
        token_lora_mask,
        M,
        top_k,
        slice_dim,
        output.stride(0),
        output.stride(1),
        output_slice.stride(0),
        output_slice.stride(1),
        slice_offset,
        BLOCK_SIZE,
    )


@torch.compile(dynamic=True)
def _align_block_size_torch(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-PyTorch align_block_size for num_experts > 1024, compiled via torch.compile."""
    device = topk_ids.device
    flat_topk_ids = topk_ids.reshape(-1).to(torch.int64)
    num_valid_tokens = flat_topk_ids.numel()
    max_total_padded_tokens = (
        (num_valid_tokens + num_experts * (block_size - 1) + block_size - 1)
        // block_size
    ) * block_size
    max_num_blocks = max_total_padded_tokens // block_size

    sorted_token_ids = torch.full(
        (max_total_padded_tokens,),
        num_valid_tokens,
        dtype=torch.int32,
        device=device,
    )
    expert_ids = torch.full(
        (max_num_blocks,),
        -1,
        dtype=torch.int32,
        device=device,
    )

    if num_valid_tokens == 0:
        num_tokens_post_padded = torch.zeros((1,), dtype=torch.int32, device=device)
        return sorted_token_ids, expert_ids, num_tokens_post_padded

    sorted_order = torch.argsort(flat_topk_ids)
    sorted_expert_ids = flat_topk_ids[sorted_order]
    expert_range = torch.arange(num_experts, device=device, dtype=torch.int64)
    counts_offsets = torch.searchsorted(sorted_expert_ids, expert_range, right=False)
    counts_end = torch.searchsorted(sorted_expert_ids, expert_range, right=True)
    counts = counts_end - counts_offsets
    padded_counts = ((counts + block_size - 1) // block_size) * block_size
    total_padded_tokens = padded_counts.sum().to(torch.int32).reshape(1)
    padded_offsets = torch.cumsum(padded_counts, dim=0) - padded_counts

    token_ranks = (
        torch.arange(num_valid_tokens, device=device, dtype=torch.int64)
        - counts_offsets[sorted_expert_ids]
    )
    output_positions = padded_offsets[sorted_expert_ids] + token_ranks
    sorted_token_ids.scatter_(
        0,
        output_positions.to(torch.int64),
        sorted_order.to(torch.int32),
    )

    block_counts = padded_counts // block_size
    actual_num_blocks = block_counts.sum()

    if max_num_blocks <= 0:
        return sorted_token_ids, expert_ids, total_padded_tokens

    block_offsets = torch.cumsum(block_counts, dim=0)
    all_block_positions = torch.arange(max_num_blocks, device=device, dtype=torch.int64)
    assigned_experts = torch.searchsorted(
        block_offsets, all_block_positions, right=True
    ).to(torch.int32)
    expert_ids.copy_(
        torch.where(
            all_block_positions < actual_num_blocks,
            assigned_experts,
            torch.full_like(assigned_experts, -1),
        )
    )

    return sorted_token_ids, expert_ids, total_padded_tokens


_align_block_size_large = _align_block_size_torch


def _merged_experts_fused_moe_lora_add_fake(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    lora_a_stacked: list[torch.Tensor],
    lora_b_stacked: list[torch.Tensor],
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    mul_routed_weight: bool,
    experts_shared_outer_loras_a: bool,
    experts_shared_outer_loras_b: bool,
    config: dict[str, Any],
    routing_cache: dict | None = None,
) -> None:
    return


def _merged_experts_fused_moe_lora_add(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    mul_routed_weight: bool,
    experts_shared_outer_loras_a: bool,
    experts_shared_outer_loras_b: bool,
    config: dict[str, Any],
    routing_cache: dict | None = None,
) -> None:
    """
    1. Prepare virtual expert routing metadata from topk_ids + token_lora_mapping * num_experts.
    2. Flatten LoRA weights from [max_loras, num_experts, ...] to [max_loras * num_experts, ...].
    3. Run regular SGLang fused-MoE kernels for LoRA A and LoRA B.
    4. Mask out tokens with token_lora_mapping == -1 on the add path.
    """
    max_loras, _, max_lora_rank, _ = lora_a.shape
    input_top_k = 1 if hidden_states.shape[0] == topk_ids.numel() else topk_ids.shape[1]

    def _merge_lora_expert_weight(t: torch.Tensor) -> torch.Tensor:
        # [max_loras, num_experts, x, y] -> [max_loras * num_experts, x, y]
        return t.reshape(t.shape[0] * t.shape[1], t.shape[2], t.shape[3])

    def _get_stage_config(
        weight: torch.Tensor,
        stage_top_k: int,
    ) -> dict[str, Any]:
        from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_config import (
            get_config_dtype_str,
            try_get_optimal_moe_config,
        )

        config_dtype = get_config_dtype_str(dtype=hidden_states.dtype)
        get_config_func = functools.partial(
            try_get_optimal_moe_config,
            weight.shape,
            weight.shape,
            stage_top_k,
            config_dtype,
        )
        try:
            cfg = get_config_func(token_lora_mapping.shape[0])
        except ValueError:
            K_dim = weight.shape[2]
            N_dim = weight.shape[1]
            if K_dim >= 1024:
                default_block_k = 256
            elif K_dim >= 64:
                default_block_k = 64
            else:
                default_block_k = max(16, K_dim)
            cfg = {
                "BLOCK_SIZE_M": min(config.get("BLOCK_SIZE_M", 64), 64),
                "BLOCK_SIZE_N": min(config.get("BLOCK_SIZE_N", 64), max(16, N_dim)),
                "BLOCK_SIZE_K": min(default_block_k, max(16, K_dim)),
                "GROUP_SIZE_M": config.get("GROUP_SIZE_M", 1),
                "num_warps": config.get("num_warps", 4),
                "num_stages": config.get("num_stages", 4),
            }
        return cfg

    def _align_block_size(
        topk_ids: torch.Tensor,
        block_size: int,
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if num_experts <= 1024:
            from sglang.srt.layers.moe.fused_moe_triton.moe_align_block_size import (
                moe_align_block_size as native_moe_align_block_size,
            )

            return native_moe_align_block_size(topk_ids, block_size, num_experts)
        return _align_block_size_large(topk_ids, block_size, num_experts)

    def _get_routing(
        topk_ids: torch.Tensor,
        token_lora_mapping: torch.Tensor,
        num_experts: int,
        shared_outer: bool,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        # Check routing_cache for cross-call reuse (gate_up and down share routing)
        cache_key = (num_experts, shared_outer, block_size)
        if routing_cache is not None:
            cached = routing_cache.get(cache_key)
            if cached is not None:
                return cached

        virtual_topk_ids, token_lora_mask, virtual_num_experts = (
            _fused_virtual_topk_ids(
                topk_ids, token_lora_mapping, num_experts, shared_outer, max_loras
            )
        )
        sorted_token_ids, expert_ids, num_tokens_post_padded = _align_block_size(
            virtual_topk_ids,
            block_size=block_size,
            num_experts=virtual_num_experts,
        )
        expert_ids = fused_sanitize_expert_ids(expert_ids, virtual_num_experts)
        result = (
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            token_lora_mask,
            virtual_num_experts,
        )

        if routing_cache is not None:
            routing_cache[cache_key] = result

        return result

    from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_kernels import (
        invoke_fused_moe_kernel,
    )

    lora_a_virtual = _merge_lora_expert_weight(lora_a)
    lora_b_virtual = _merge_lora_expert_weight(lora_b)
    num_experts_a = lora_a.shape[1]
    num_experts_b = lora_b.shape[1]

    intermediate = torch.empty(
        [token_lora_mapping.shape[0], topk_ids.shape[1], max_lora_rank],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    a_stage_config = _get_stage_config(lora_a_virtual, input_top_k)
    (
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        token_lora_mask,
        virtual_num_experts,
    ) = _get_routing(
        topk_ids,
        token_lora_mapping,
        num_experts_a,
        experts_shared_outer_loras_a,
        a_stage_config["BLOCK_SIZE_M"],
    )

    invoke_fused_moe_kernel(
        hidden_states,
        lora_a_virtual,
        None,
        intermediate,
        None,
        None,
        None,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        False,
        input_top_k,
        a_stage_config,
        tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16,
        False,
        False,
        False,
        False,
        False,
        None,
    )

    b_stage_config = _get_stage_config(lora_b_virtual, 1)
    (
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        token_lora_mask,
        virtual_num_experts,
    ) = _get_routing(
        topk_ids,
        token_lora_mapping,
        num_experts_b,
        experts_shared_outer_loras_b,
        b_stage_config["BLOCK_SIZE_M"],
    )

    invoke_fused_moe_kernel(
        intermediate.view(-1, max_lora_rank),
        lora_b_virtual,
        None,
        output,
        None,
        None,
        None,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight,
        1,
        b_stage_config,
        tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16,
        False,
        False,
        False,
        False,
        False,
        None,
        fuse_add_to_output=True,
        add_output_mask=token_lora_mask,
        router_topk=topk_ids.shape[1],
    )


# Cannot register as a torch custom op because dict[str, Any] params are
# not supported by torch.library.infer_schema. Export the implementation directly.
merged_experts_fused_moe_lora_add = _merged_experts_fused_moe_lora_add
