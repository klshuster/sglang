"""
Usage:
cd test/srt
python3 -m unittest test_qwen3_next_deterministic.TestFlashInferDeterministic
"""

import unittest

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_deterministic_utils import (
    COMMON_SERVER_ARGS,
    TestDeterministicBase,
)
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST

register_cuda_ci(est_time=240, suite="nightly-4-gpu", nightly=True)

QWEN3_NEXT = "Qwen/Qwen3-Next-80B-A3B-Instruct"


class TestFlashInferDeterministic(TestDeterministicBase):
    @classmethod
    def get_model(cls):
        return QWEN3_NEXT

    # Test with flashinfer attention backend
    @classmethod
    def get_server_args(cls):
        args = COMMON_SERVER_ARGS
        args.extend(["--attention-backend", "flashinfer", "--tp", "4"])
        return args


class TestTritonDeterministic(TestDeterministicBase):
    @classmethod
    def get_model(cls):
        return QWEN3_NEXT

    # Test with triton attention backend
    @classmethod
    def get_server_args(cls):
        args = COMMON_SERVER_ARGS
        args.extend(["--attention-backend", "triton", "--tp", "4"])
        return args

    # Radix "tips" past the prompt boundary are decode-program mamba state;
    # deterministic inference must never serve a prefix hit from them.
    def test_no_prefix_hit_into_generated_tokens(self):
        prompt_len = 64
        # crosses --mamba-track-interval (256) during decode
        first_max_new, second_max_new = 256, 32
        prompt_ids = list(range(5, 5 + prompt_len))
        url = DEFAULT_URL_FOR_TEST
        requests.post(url + "/flush_cache").raise_for_status()

        def generate(input_ids, max_new_tokens):
            response = requests.post(
                url + "/generate",
                json={
                    "input_ids": input_ids,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": max_new_tokens,
                        "ignore_eos": True,
                    },
                },
            )
            response.raise_for_status()
            return response.json()

        first = generate(prompt_ids, first_max_new)
        self.assertEqual(len(first["output_ids"]), first_max_new)
        second = generate(prompt_ids + first["output_ids"], second_max_new)
        self.assertLessEqual(second["meta_info"]["cached_tokens"], prompt_len)



if __name__ == "__main__":
    unittest.main()
