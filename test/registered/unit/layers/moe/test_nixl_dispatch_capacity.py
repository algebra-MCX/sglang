"""NIXL EP passes each forward's dispatch capacity to the RDMA buffer layout,
so every EP rank must derive the same value from DP-synchronized counts."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import sglang.srt.layers.moe.token_dispatcher.nixl as nixl
import sglang.srt.utils.common as common
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestNixlDispatchCapacity(CustomTestCase):
    def _capacity(self, global_num_tokens):
        impl = SimpleNamespace(num_max_dispatch_tokens_per_rank=128)
        with patch.object(
            nixl, "get_dp_global_num_tokens", return_value=global_num_tokens
        ):
            return nixl._NixlEPDispatcherImpl._dispatch_capacity(impl)

    def test_capacity_is_aligned_dp_max_bounded_by_reservation(self):
        self.assertEqual(self._capacity([3, 70]), 72)
        self.assertEqual(self._capacity([1000, 1]), 128)
        # TBO sub-batches publish no counts and keep the static reservation.
        self.assertEqual(self._capacity(None), 128)

    def test_nixl_forces_dp_synchronized_counts(self):
        # With a dense TP of 1 and a DP LM head, other a2a backends keep
        # rank-local counts; NIXL needs them synchronized to agree on capacity.
        parallel = SimpleNamespace(
            attn_dp_enabled=True,
            num_dp_ranks=8,
            tp_size=8,
            moe_dense_tp_size=1,
            enable_dp_lm_head=True,
        )
        exec_ = SimpleNamespace(moe=SimpleNamespace(elastic_ep_backend=None))
        with (
            patch.object(common, "get_parallel", return_value=parallel),
            patch.object(common, "get_exec", return_value=exec_),
        ):
            self.assertFalse(common.require_mlp_tp_gather(moe_a2a_backend="deepep"))
            self.assertTrue(common.require_mlp_tp_gather(moe_a2a_backend="nixl"))


if __name__ == "__main__":
    unittest.main()
