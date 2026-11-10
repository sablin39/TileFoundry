"""Dist-MoE BF16 scratch reuse and deferred weight-gradient counterexamples."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from tilefoundry.analysis import BufferAccess, BufferLifetime, check_buffer_lifetimes


def backward_lifetimes(policy, rows=128, width=2048):
    """Describe one expert's h2/grad_h1 reuse with illustrative dependency ticks."""
    size = rows * width * 2
    hidden = BufferLifetime(
        "hidden_activation",
        "expert.scratch",
        0,
        size,
        BufferAccess("forward", 0, 1),
        (BufferAccess("down_weight_gradient", 3, 6),),
    )
    gradient = BufferLifetime(
        "gate_up_activation_gradient",
        "expert.scratch",
        0,
        2 * size,
        BufferAccess("swiglu_backward", 6, 7),
        (
            BufferAccess("gate_up_input_gradient", 7, 9),
            BufferAccess("gate_up_weight_gradient", 9, 12),
        ),
    )
    if policy == "baseline":
        return hidden, gradient
    gradient = BufferLifetime(
        gradient.value,
        gradient.allocation,
        0,
        2 * size,
        BufferAccess("swiglu_backward", 4, 5),
        (
            BufferAccess("gate_up_input_gradient", 5, 7),
            BufferAccess("gate_up_weight_gradient", 10, 13),
        ),
    )
    if policy == "unsafe_defer":
        hidden = BufferLifetime(
            hidden.value,
            hidden.allocation,
            0,
            size,
            hidden.producer,
            (BufferAccess("down_weight_gradient", 7, 10),),
        )
        return hidden, gradient
    if policy != "preserve_and_defer":
        raise ValueError(policy)
    preserve = BufferAccess("preserve_hidden", 1, 2)
    return (
        BufferLifetime(hidden.value, hidden.allocation, 0, size, hidden.producer, (preserve,)),
        BufferLifetime(
            "preserved_hidden",
            "expert.saved_hidden",
            0,
            size,
            preserve,
            (BufferAccess("down_weight_gradient", 7, 10),),
        ),
        gradient,
    )


def check():
    result = {}
    for policy in ("baseline", "unsafe_defer", "preserve_and_defer"):
        values = backward_lifetimes(policy)
        hazards = check_buffer_lifetimes(values)
        assert bool(hazards) == (policy == "unsafe_defer")
        result[policy] = {
            "lifetimes": [asdict(v) for v in values],
            "hazards": [asdict(h) for h in hazards],
        }
    result["decision"] = {
        "candidate": "preserve_and_defer",
        "question": "Can input gradients reach the preceding stage earlier?",
        "extra_bytes_per_routed_row": 2048 * 2,
        "required_measurements": ["input_gradient_delivery_us", "backward_completion_us"],
        "time_unit": "dependency ticks; no hardware rates",
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = check()
    rendered = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n")
    print(rendered)
