"""Whole-layer equations, complete VJPs, masks, and authored TileFoundry HIR."""

import pytest
import torch
from types import SimpleNamespace

from .full_layer import (
    boundary_policy,
    full_layer_graph,
    reference_layer,
    synthetic_inputs,
)
from .graph import evaluate_graph
from .lower import check_program, deep_ir_visitors, execute, function, lower
from .planner import plan_from_saved


@pytest.mark.parametrize("batch", [1, 2])
def test_pinned_official_complete_layer_padding_and_all_gradients(batch):
    from .full_layer import load_official_layer

    config = {**tiny_config(), "layer_types": ["linear_attention"]}
    graph = full_layer_graph(config, 7, batch=batch, chunk_size=4)
    values = synthetic_inputs(graph)
    block = load_official_layer()(SimpleNamespace(**config), layer_idx=0).float()
    with torch.no_grad():
        for name, parameter in block.named_parameters():
            parameter.copy_(values[name])
    hidden = values["hidden_states"]
    mask = values["attention_mask"]
    expected = block(hidden, attention_mask=mask.squeeze(-1))
    cot = torch.randn_like(expected)
    official_args = {"hidden_states": hidden, **dict(block.named_parameters())}
    names = graph.metadata["gradient_inputs"]
    official_grad = torch.autograd.grad(
        expected, [official_args[n] for n in names], cot
    )
    reference = reference_layer(values, config)
    reference_grad = torch.autograd.grad(reference, [values[n] for n in names], cot)
    for x, y in zip(
        (expected, *official_grad), (reference, *reference_grad), strict=True
    ):
        torch.testing.assert_close(x, y, rtol=3e-5, atol=3e-5)
    assert torch.count_nonzero(official_grad[0][0, -2:]) == 0


def tiny_config(group=2, taps=4):
    return dict(
        hidden_size=12,
        linear_num_key_heads=2,
        linear_num_value_heads=2 * group,
        linear_key_head_dim=4,
        linear_value_head_dim=3,
        linear_conv_kernel_dim=taps,
        rms_norm_eps=1e-6,
        hidden_act="silu",
    )


@pytest.mark.parametrize("length,group,taps", [(1, 2, 4), (7, 2, 4), (9, 3, 1)])
@pytest.mark.parametrize("save", [False, True])
def test_full_layer_all_gradients(length, group, taps, save):
    config = tiny_config(group, taps)
    graph = full_layer_graph(config, length, batch=2, chunk_size=4)
    tensors = synthetic_inputs(graph, dtype=torch.float64)
    expected = reference_layer(tensors, config)
    cot = torch.randn_like(expected)
    grads = torch.autograd.grad(
        expected, [tensors[n] for n in graph.metadata["gradient_inputs"]], cot
    )
    plan = plan_from_saved(graph, boundary_policy(graph) if save else [])
    inputs = {i: tensors[graph.nodes[i].name].detach() for i in graph.forward_inputs}
    outputs, _ = evaluate_graph(graph, [*graph.forward_outputs, *plan.saved], inputs)
    inputs.update(zip([*graph.forward_outputs, *plan.saved], outputs, strict=True))
    inputs[graph.backward_inputs[0]] = cot
    actual_grads, _ = evaluate_graph(
        graph, graph.backward_outputs, inputs, recipes=plan.recipes
    )
    for actual, wanted in zip(
        (outputs[0], *actual_grads), (expected, *grads), strict=True
    ):
        torch.testing.assert_close(actual, wanted, rtol=2e-9, atol=2e-9)
    if length > 2:
        assert torch.count_nonzero(actual_grads[0][0, -2:]) == 0


@pytest.mark.parametrize("save", [False, True])
def test_entire_layer_authored_hir(tmp_path, save):
    config = tiny_config()
    graph = full_layer_graph(config, 7, batch=2, chunk_size=4)
    tensors = synthetic_inputs(graph)
    expected = reference_layer(tensors, config)
    cot = torch.randn_like(expected)
    grads = torch.autograd.grad(
        expected, [tensors[n] for n in graph.metadata["gradient_inputs"]], cot
    )
    plan = plan_from_saved(graph, boundary_policy(graph) if save else [])
    program = lower(graph, plan, tmp_path / f"full_{save}.py")
    with deep_ir_visitors():
        for phase in ("forward", "backward"):
            check_program(program.module, function(program.module, phase))
    inputs = {i: tensors[graph.nodes[i].name].detach() for i in graph.forward_inputs}
    inputs[graph.backward_inputs[0]] = cot
    outputs, actual_grads = execute(program, inputs)
    for actual, wanted in zip(
        (*outputs, *actual_grads), (expected, *grads), strict=True
    ):
        torch.testing.assert_close(actual, wanted, rtol=3e-5, atol=3e-5)
    source = program.path.read_text()
    assert "import torch" not in source
    assert "tf.matmul" in source and "tf.silu" in source and "tf.rsqrt" in source
    assert len(program.backward.outputs) == 10


def test_full_layer_rejects_unsupported_activation():
    config = tiny_config()
    config["hidden_act"] = "gelu"
    with pytest.raises(ValueError, match="SiLU"):
        full_layer_graph(config, 7)
