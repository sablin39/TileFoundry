"""Independent CPU oracles and explicit numerical validation contracts."""

import torch

from .problems import recurrent_oracle


def sample(graph, seed=0, case="random", dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)

    def randn(shape):
        return torch.randn(*shape, dtype=dtype, generator=generator)

    args = [randn(graph.nodes[i].shape) for i in graph.forward_inputs]
    if graph.name == "chunk_gated_delta_rule":
        q, k, v, gate, beta, state = args
        k = k / k.norm(dim=1, keepdim=True).clamp_min(1e-10)
        gate = -torch.rand(gate.shape, dtype=dtype, generator=generator) * 0.5
        beta = torch.rand(beta.shape, dtype=dtype, generator=generator)
        if case == "zero_decay":
            gate.zero_()
        if case == "weak_decay":
            gate.mul_(0.002)
        if case == "strong_decay":
            gate.fill_(-30.0)
        if case == "zero_beta":
            beta.zero_()
        if case == "unit_beta":
            beta.fill_(1.0)
        if case == "zero_keys":
            k.zero_()
        if case == "zero_initial":
            state.zero_()
        args = [q, k, v, gate, beta, state]
    elif case == "relu_zero":
        args[1].zero_()
    args = [a.detach().requires_grad_() for a in args]
    cotangents = [randn(graph.nodes[i].shape) for i in graph.backward_inputs]
    if case == "state_only":
        cotangents[0].zero_()
    if case == "output_only":
        cotangents[1].zero_()
    return args, cotangents


def reference(graph, args):
    if graph.name == "chunk_gated_delta_rule":
        return recurrent_oracle(*args)
    h = torch.relu(args[0] @ args[1])
    if graph.name == "relu_sum":
        return (h.sum().reshape(1, 1),)
    return (h @ args[2],)


def oracle(graph, args, cotangents):
    output = reference(graph, args)
    gradients = torch.autograd.grad(output, args, grad_outputs=cotangents)
    return output, gradients


def assert_results(actual, expected, *, dtype=torch.float32):
    tolerance = 2e-9 if dtype == torch.float64 else 2e-5
    errors = []
    for x, y in zip(actual, expected, strict=True):
        if not torch.isfinite(x).all():
            raise AssertionError("Generated formula produced nonfinite values")
        torch.testing.assert_close(x, y, rtol=tolerance, atol=tolerance)
        errors.append((x - y).abs().max().item())
    return errors
