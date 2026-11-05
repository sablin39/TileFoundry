"""Keep native call checking linear in shared graph nodes, not expanded trees."""

from tilefoundry.analysis import check_program
from tilefoundry.dsl import Tensor, Topology, func, module, tf
from tilefoundry.ir.core import Call
from tilefoundry_nvidia import CudaTarget


def test_call_checker_does_not_structurally_hash_shared_expressions(monkeypatch):
    @module(
        entry="forward",
        target=CudaTarget("nvidia.h200_sxm"),
        topologies=(Topology("cta", 1),),
    )
    class Shared:
        @func
        def forward(x: Tensor[(2, 2), "f32"]):  # noqa: F821 - TileFoundry dtype syntax
            a = tf.mul(x, x)
            b = tf.add(a, a)
            c = tf.mul(b, b)
            return tf.add(c, c)

    def structural_hash_is_a_regression(self):
        raise AssertionError(
            "The checker must not structurally hash the expression DAG"
        )

    monkeypatch.setattr(Call, "__hash__", structural_hash_is_a_regression)
    check_program(Shared, Shared.functions[0])
