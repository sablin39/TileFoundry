"""Extents stated by the caller reach the functions that declare them.

A dimension no argument shape pins -- one only a callee allocates with -- is
bound at evaluation from the supplied mapping, in the entry's own context and
in every callee's, across module boundaries. A dimension the argument shapes
also pin has to agree with them, and one nothing was given for still says so.
"""

from __future__ import annotations

import pytest
import torch

from tests.fixtures.logical.callee_dim import BT, Rooted, Staged
from tilefoundry.dsl import *
from tilefoundry.evaluator import evaluate
from tilefoundry.evaluator.value import EvalError
from tilefoundry.runtime import DictResource


def test_a_callee_only_dim_binds_from_the_supplied_mapping():
    x = torch.randn(4)
    assert torch.allclose(evaluate(Staged.entry_function(), x, dims={"BT": 8}), x)


def test_the_supplied_mapping_crosses_module_boundaries():
    x = torch.randn(4)
    assert torch.allclose(evaluate(Rooted.load(DictResource({})), x, dims={"BT": 8}), x)


def test_a_dim_nothing_supplied_still_says_so():
    with pytest.raises(EvalError, match=r"unbound DimVar 'BT'"):
        evaluate(Staged.entry_function(), torch.randn(4))


@func
def _echo(x: Tensor[(BT,), "f32"]) -> Tensor[(BT,), "f32"]:
    return x


def test_a_supplied_extent_must_agree_with_the_argument_shapes():
    with pytest.raises(EvalError, match=r"supplied as 4 but the arguments bind it to 8"):
        evaluate(_echo, torch.randn(8), dims={"BT": 4})


def test_agreeing_supplied_and_argument_extents_bind():
    x = torch.randn(8)
    assert torch.allclose(evaluate(_echo, x, dims={"BT": 8}), x)
