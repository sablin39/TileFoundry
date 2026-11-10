"""Measure lifetime-guided expert-backward scheduling on two SM103 NVLink peers."""

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
from pathlib import Path

import tilelang
import torch
from cuda.bindings import runtime
from kernels import swiglu_backward
from lifetime import backward_lifetimes, check

from tilefoundry.analysis import check_buffer_lifetimes


class Experiment:
    def __init__(self, rows, hidden, expert):
        self.rows, self.hidden, self.expert = rows, hidden, expert
        assert not check_buffer_lifetimes(backward_lifetimes("preserve_and_defer", rows, expert))
        torch.cuda.set_device(0)
        torch.manual_seed(671)
        self.x = torch.randn(rows, hidden, device="cuda:0", dtype=torch.bfloat16)
        self.w13 = torch.randn(
            hidden, 2 * expert, device="cuda:0", dtype=torch.bfloat16
        ) / math.sqrt(hidden)
        self.w2 = torch.randn(expert, hidden, device="cuda:0", dtype=torch.bfloat16) / math.sqrt(
            expert
        )
        self.dy = torch.randn_like(self.x) / math.sqrt(hidden)
        self.gate_up = self.x @ self.w13
        gate, up = self.gate_up.float().chunk(2, dim=1)
        self.h2 = (torch.nn.functional.silu(gate) * up).bfloat16()
        self.scratch = torch.empty(rows * 2 * expert, device="cuda:0", dtype=torch.bfloat16)
        self.hidden_view = self.scratch[: rows * expert].view(rows, expert)
        self.grad_view = self.scratch.view(rows, 2 * expert)
        self.saved_hidden = torch.empty_like(self.h2)
        self.dh = torch.empty_like(self.h2)
        self.dx = torch.empty_like(self.x)
        self.dw13, self.dw2 = torch.empty_like(self.w13), torch.empty_like(self.w2)
        self.received = torch.empty_like(self.x, device="cuda:1")
        self.compute = torch.cuda.Stream(device=0)
        self.transfer = torch.cuda.Stream(device=0)
        (error,) = runtime.cudaDeviceEnablePeerAccess(1, 0)
        if error not in (
            runtime.cudaError_t.cudaSuccess,
            runtime.cudaError_t.cudaErrorPeerAccessAlreadyEnabled,
        ):
            raise RuntimeError(error)
        self.kernel = swiglu_backward(rows, expert)
        self.start = torch.cuda.Event(enable_timing=True, external=True)
        self.produced = torch.cuda.Event()
        self.copied = torch.cuda.Event()
        self.ready = torch.cuda.Event(enable_timing=True, external=True)
        self.computed = torch.cuda.Event(enable_timing=True, external=True)
        self.end = torch.cuda.Event(enable_timing=True, external=True)
        self.graphs = {}
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)

    def enqueue(self, policy):
        with torch.cuda.stream(self.compute):
            self.start.record()
            if policy == "preserve_and_defer":
                self.saved_hidden.copy_(self.hidden_view)
            torch.mm(self.dy, self.w2.T, out=self.dh)
            if policy == "baseline":
                torch.mm(self.hidden_view.T, self.dy, out=self.dw2)
            self.kernel(self.gate_up, self.dh, self.grad_view)
            torch.mm(self.grad_view, self.w13.T, out=self.dx)
            self.produced.record()
        with torch.cuda.stream(self.transfer):
            self.transfer.wait_event(self.produced)
            (error,) = runtime.cudaMemcpyAsync(
                self.received.data_ptr(),
                self.dx.data_ptr(),
                self.dx.numel() * self.dx.element_size(),
                runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                self.transfer.cuda_stream,
            )
            if error != runtime.cudaError_t.cudaSuccess:
                raise RuntimeError(error)
            self.ready.record()
            self.copied.record()
        with torch.cuda.stream(self.compute):
            if policy == "preserve_and_defer":
                torch.mm(self.saved_hidden.T, self.dy, out=self.dw2)
            torch.mm(self.x.T, self.grad_view, out=self.dw13)
            self.computed.record()
            self.compute.wait_event(self.copied)
            self.end.record()

    def run(self, policy):
        # Both paths start with the same forward-produced h2 in the aliased region.
        self.hidden_view.copy_(self.h2)
        self.compute.wait_stream(torch.cuda.current_stream(0))
        with torch.cuda.stream(self.compute):
            if policy in self.graphs:
                self.graphs[policy].replay()
            else:
                self.enqueue(policy)
        self.end.synchronize()
        self.ready.synchronize()
        delivery = self.start.elapsed_time(self.ready) * 1000
        compute = self.start.elapsed_time(self.computed) * 1000
        return {
            "input_gradient_delivery_us": delivery,
            "backward_completion_us": max(delivery, compute),
            "joined_graph_completion_us": self.start.elapsed_time(self.end) * 1000,
        }

    def capture(self):
        for policy in ("baseline", "preserve_and_defer"):
            for _ in range(3):
                self.run(policy)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self.compute):
                self.enqueue(policy)
            self.graphs[policy] = graph

    def validate(self):
        x, w13, w2 = (t.detach().clone().requires_grad_() for t in (self.x, self.w13, self.w2))
        gate, up = (x @ w13).float().chunk(2, dim=1)
        y = (torch.nn.functional.silu(gate) * up).bfloat16() @ w2
        expected = torch.autograd.grad((y.float() * self.dy.float()).sum(), (x, w13, w2))
        errors = {}
        for policy in ("baseline", "preserve_and_defer"):
            for repeat in range(3):
                self.run(policy)
                actual = (self.dx, self.dw13, self.dw2)
                for name, value, reference in zip(
                    ("input", "gate_up_weight", "down_weight"), actual, expected
                ):
                    torch.testing.assert_close(value, reference, atol=0.004, rtol=0.016)
                    errors[f"{policy}.{name}"] = (
                        (value.float() - reference.float()).abs().max().item()
                    )
                torch.testing.assert_close(self.received.cpu(), self.dx.cpu(), atol=0, rtol=0)
        return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trace", type=Path)
    args = parser.parse_args()
    if args.repeats < 10 or any(rows < 1 for rows in args.rows):
        parser.error("use positive row counts and at least ten timing repeats")
    assert torch.cuda.device_count() >= 2
    assert all(torch.cuda.get_device_capability(i) == (10, 3) for i in (0, 1))
    assert torch.cuda.can_device_access_peer(0, 1)
    topology = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True)
    report = {
        "schema": 1,
        "model": check(),
        "cc": [10, 3],
        "torch": torch.__version__,
        "tilelang": tilelang.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "topology": topology,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        "scope": "expert backward plus peer-address cudaMemcpyAsync; CUDA-graph replay with CUDA-event timing",
        "completion_definition": "max(last weight-gradient completion, peer-copy completion); graph join reported separately",
        "input_policy": "same forward-produced h2 restored before timing; candidate preservation copy timed",
        "objective": {"delivery_improvement_min": 0.1, "completion_regression_max": 0.1},
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path(__file__).with_name("kernels.py"),
                Path(__file__).with_name("lifetime.py"),
            )
        },
        "validation": [],
        "cases": [],
    }
    for rows, hidden, expert in [(7, 64, 32), (129, 256, 128)]:
        trial = Experiment(rows, hidden, expert)
        trial.validate()
        trial.capture()
        report["validation"].append({"shape": [rows, hidden, expert], "errors": trial.validate()})
    for rows in args.rows:
        trial = Experiment(rows, 7168, 2048)
        trial.validate()
        trial.capture()
        errors = trial.validate()
        for _ in range(5):
            for policy in ("baseline", "preserve_and_defer"):
                trial.run(policy)
        values = {p: [] for p in ("baseline", "preserve_and_defer")}
        for repeat in range(args.repeats):
            for policy in tuple(values)[:: 1 if repeat % 2 == 0 else -1]:
                values[policy].append(trial.run(policy))
        summary = {
            p: {
                key: {
                    "median": statistics.median(v[key] for v in samples),
                    "p10": statistics.quantiles(
                        [v[key] for v in samples], n=10, method="inclusive"
                    )[0],
                    "p90": statistics.quantiles(
                        [v[key] for v in samples], n=10, method="inclusive"
                    )[-1],
                    "min": min(v[key] for v in samples),
                    "max": max(v[key] for v in samples),
                }
                for key in samples[0]
            }
            for p, samples in values.items()
        }
        baseline, candidate = (summary[p] for p in ("baseline", "preserve_and_defer"))
        improvement = (
            1
            - candidate["input_gradient_delivery_us"]["median"]
            / baseline["input_gradient_delivery_us"]["median"]
        )
        regression = (
            candidate["backward_completion_us"]["median"]
            / baseline["backward_completion_us"]["median"]
            - 1
        )
        separated = (
            candidate["input_gradient_delivery_us"]["p90"]
            < baseline["input_gradient_delivery_us"]["p10"]
        )
        report["cases"].append(
            {
                "rows": rows,
                "hidden": 7168,
                "expert": 2048,
                "extra_live_bytes": rows * 2048 * 2,
                "errors": errors,
                "decision": "candidate meets delivery objective"
                if improvement >= 0.1 and regression <= 0.1 and separated
                else "retain baseline",
                "delivery_improvement": improvement,
                "completion_regression": regression,
                "samples_us": values,
                "summary_us": summary,
            }
        )
        print(rows, json.dumps(summary), flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    source = args.out.with_suffix(".cu")
    source.write_text(trial.kernel.get_kernel_source())
    if args.trace:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as profile:
            for policy in ("baseline", "preserve_and_defer"):
                with torch.profiler.record_function(policy):
                    trial.run(policy)
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        profile.export_chrome_trace(str(args.trace))


if __name__ == "__main__":
    main()
