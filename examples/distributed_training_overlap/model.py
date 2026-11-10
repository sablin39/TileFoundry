"""Small residual routed FFN with separately executable input and weight gradients."""

import torch

from tilefoundry.analysis import check_program
from tilefoundry.dsl import Tensor, func, module, tf
from tilefoundry.evaluator import evaluate
from tilefoundry.target import CpuTarget

ROWS, HIDDEN, EXPERT = 4, 8, 16


@module(entry="forward", target=CpuTarget())
class Expert:
    @func
    def forward(
        x: Tensor[(ROWS, HIDDEN), "f32"],
        wg: Tensor[(HIDDEN, EXPERT), "f32"],
        wu: Tensor[(HIDDEN, EXPERT), "f32"],
        wd: Tensor[(EXPERT, HIDDEN), "f32"],
    ):
        gate = tf.matmul(x, wg)
        up = tf.matmul(x, wu)
        hidden = tf.silu(gate) * up
        return gate, up, hidden, tf.matmul(hidden, wd)

    @func
    def input_gradient(
        dy: Tensor[(ROWS, HIDDEN), "f32"],
        gate: Tensor[(ROWS, EXPERT), "f32"],
        up: Tensor[(ROWS, EXPERT), "f32"],
        wg: Tensor[(HIDDEN, EXPERT), "f32"],
        wu: Tensor[(HIDDEN, EXPERT), "f32"],
        wd: Tensor[(EXPERT, HIDDEN), "f32"],
    ):
        dh = tf.matmul(dy, wd, b_layout="NK")
        sigmoid = tf.sigmoid(gate)
        dg = dh * up * sigmoid * (1.0 + gate * (1.0 - sigmoid))
        du = dh * gate * sigmoid
        return tf.matmul(dg, wg, b_layout="NK") + tf.matmul(du, wu, b_layout="NK"), dg, du

    @func
    def weight_gradient(
        x: Tensor[(ROWS, HIDDEN), "f32"],
        h: Tensor[(ROWS, EXPERT), "f32"],
        dg: Tensor[(ROWS, EXPERT), "f32"],
        du: Tensor[(ROWS, EXPERT), "f32"],
        dy: Tensor[(ROWS, HIDDEN), "f32"],
    ):
        return (
            tf.matmul(x, dg, a_layout="KM"),
            tf.matmul(x, du, a_layout="KM"),
            tf.matmul(h, dy, a_layout="KM"),
        )


class TrainingReference:
    def __init__(self, stages, microbatches):
        torch.manual_seed(671)
        self.functions = {f.name: check_program(Expert, f)[1] for f in Expert.functions}
        self.weights = [
            [
                tuple(
                    (torch.randn(*shape) * 0.15).requires_grad_()
                    for shape in ((HIDDEN, EXPERT), (HIDDEN, EXPERT), (EXPERT, HIDDEN))
                )
                for _ in range(5)
            ]
            for _ in range(stages)
        ]
        self.inputs = [torch.randn(ROWS, HIDDEN, requires_grad=True) for _ in range(microbatches)]
        self.seeds = [torch.randn(ROWS, HIDDEN) / microbatches for _ in self.inputs]
        self.outputs, self.saved, self.dx, self.grads = {}, {}, {}, {}
        self.expected_outputs = []
        for x in self.inputs:
            for stage in range(stages):
                out = x
                for expert, (wg, wu, wd) in enumerate(self.weights[stage]):
                    h = torch.nn.functional.silu(x @ wg) * (x @ wu)
                    out = out + 0.1 * self.score(stage, expert) * (h @ wd)
                x = out
            self.expected_outputs.append(x)
        leaves = [w for stage in self.weights for expert in stage for w in expert]
        loss = sum((x * dy).sum() for x, dy in zip(self.expected_outputs, self.seeds))
        self.expected = torch.autograd.grad(loss, (*self.inputs, *leaves))

    @staticmethod
    def score(stage, expert):
        if expert == 4:
            return torch.ones(ROWS, 1)  # shared expert
        token = torch.arange(ROWS).reshape(-1, 1)
        return ((token + stage) % 4 == expert).float() * 0.35 + (
            (token + stage + 1) % 4 == expert
        ).float() * 0.65

    def call(self, name, *args):
        return evaluate(self.functions[name], *args)

    def execute(self, action):
        kind, direction, batch, stage = action
        key = direction, batch, stage
        with torch.no_grad():
            if kind == "fwd":
                x = (
                    self.inputs[batch].detach()
                    if stage == 0
                    else self.outputs[direction, batch, stage - 1]
                )
                out, state = x.clone(), []
                for expert, weights in enumerate(self.weights[stage]):
                    gate, up, h, y = self.call("forward", x, *weights)
                    out += 0.1 * self.score(stage, expert) * y
                    state.append({"x": x, "gate": gate, "up": up, "h": h})
                self.outputs[key], self.saved[key] = out, state
            elif kind == "bwd_input":
                dy = (
                    self.seeds[batch]
                    if stage == len(self.weights) - 1
                    else self.dx[direction, batch, stage + 1]
                )
                dx = dy.clone()
                for expert, (state, weights) in enumerate(
                    zip(self.saved[key], self.weights[stage])
                ):
                    state["dy"] = dy * (0.1 * self.score(stage, expert))
                    local, state["dg"], state["du"] = self.call(
                        "input_gradient", state["dy"], state["gate"], state["up"], *weights
                    )
                    dx += local
                self.dx[key] = dx
            else:
                for expert, state in enumerate(self.saved.pop(key)):
                    gradients = self.call(
                        "weight_gradient", *(state[n] for n in ("x", "h", "dg", "du", "dy"))
                    )
                    for index, grad in enumerate(gradients):
                        grad_key = direction, stage, expert, index
                        self.grads[grad_key] = self.grads.get(grad_key, 0) + grad

    def validate(self):
        errors = []

        def compare(actual, expected):
            torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
            errors.append((actual - expected).abs().max().item())

        batches = len(self.inputs)
        for batch in range(batches):
            direction = int(batch >= batches // 2)
            compare(
                self.outputs[direction, batch, len(self.weights) - 1], self.expected_outputs[batch]
            )
            compare(self.dx[direction, batch, 0], self.expected[batch])
        expected = iter(self.expected[batches:])
        for stage in range(len(self.weights)):
            for expert in range(5):
                for index in range(3):
                    # The two counter-flow parameter replicas contribute to the same logical weight.
                    compare(
                        sum(self.grads[d, stage, expert, index] for d in (0, 1)), next(expected)
                    )
        assert not self.saved
        return max(errors)
