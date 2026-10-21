"""A stated dimension is validated against what the reachable program declares.

The entry's own signature is not the whole namespace: a helper can introduce
a dimension on its own scratch, in the same module or across a boundary, and
analysis binds it. A name nothing reachable declares is refused with the
functions that do declare dims -- a binding the block dropped declared
nothing.
"""

from __future__ import annotations

import pytest

from tests.fixtures.logical.callee_dim import DeadScratch, Rooted, Staged
from tilefoundry.analysis.check import check_program
from tilefoundry.ir.hir.specialize import SpecializationError


def test_a_dim_only_a_callee_declares_binds():
    check_program(Staged, Staged.entry_function(), dims={"BT": 8})


def test_a_dim_only_a_child_module_declares_binds():
    check_program(Rooted, Rooted.entry_function(), dims={"BT": 8})


def test_an_unknown_name_meets_the_functions_that_declare_dims():
    with pytest.raises(
        SpecializationError,
        match=r"has no dimension named \['BTT'\]; the dimensions it reaches are BT \(stage\)",
    ):
        check_program(Staged, Staged.entry_function(), dims={"BTT": 8})


def test_a_binding_the_block_dropped_declares_nothing():
    with pytest.raises(
        SpecializationError, match=r"no function it reaches declares a dimension"
    ):
        check_program(DeadScratch, DeadScratch.entry_function(), dims={"BT": 8})
