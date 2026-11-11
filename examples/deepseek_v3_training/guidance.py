"""Compare training-layout payloads and bounded route schedules independently."""

import argparse
import json
from math import isfinite
from pathlib import Path

from visualize import PROBES, read_samples
from workload import CONTEXT, HIDDEN, PASSES, TOP_K, cp_payload, expert_ids, trace


def context_layouts(ranks):
    """Separate compact-KV representation from the collective used to move it."""
    payload = cp_payload(ranks)
    ulysses = sum(payload["Ulysses"])
    ring_forward, ring_gradient = payload["Ring"]
    return {
        "cp_gpus": ranks,
        "scope": "logical BF16 bytes sent per GPU, per sequence, per MLA layer",
        "ulysses": {"fwd_bytes": ulysses, "bwd_bytes": ulysses},
        "compact_ring": {
            "fwd_bytes": ring_forward,
            "bwd_bytes": ring_forward + ring_gradient,
        },
        "compact_all_gather": {
            "fwd_bytes": ring_forward,
            "bwd_bytes": ring_forward,
            "retained_compact_input_bytes_per_gpu": 2 * CONTEXT * (512 + 64),
        },
        "ring_compact_block_bytes": 2 * (CONTEXT // ranks) * (512 + 64),
        "fwd_block_projections_across_group": {
            "ulysses": ranks,
            "compact_ring": ranks * (ranks + 1) // 2,
        },
    }


def expert_route():
    """Count the same token's assignments before and after destination grouping."""
    selected = expert_ids(0, "colocated")
    owners = sorted({expert // TOP_K for expert in selected})
    row_bytes = HIDDEN * 2
    return {
        "scope": "one source token selecting eight experts on two remote GPUs",
        "source_gpu": 0,
        "expert_owners": owners,
        "expert_assignments": len(selected),
        "destination_rows": len(owners),
        "naive_dispatch_bytes": len(selected) * row_bytes,
        "compact_dispatch_bytes": len(owners) * row_bytes,
        "naive_return_bytes": len(selected) * row_bytes,
        "compact_return_bytes": len(owners) * row_bytes,
        "peer_access": {
            "published_source_rows": 1,
            "addressed_input_row_reads": len(selected),
            "addressed_output_row_writes": len(selected),
        },
        "next_measurement": "destination grouping, peer-read reuse and expert row imbalance",
    }


def route_schedules(tiles, slots, rows, samples):
    """Compare identical work with stage barriers or bounded chunk readiness."""
    result = {}
    for name, phases in PASSES.items():
        variants = []
        for policy, count in [("bulk", tiles)] + [("tile_ready", n) for n in slots]:
            intervals = trace(phases, tiles, count, policy, samples)
            end = max(event.end for event in intervals)
            busy = {
                resource: sum(
                    event.end - event.start
                    for event in intervals
                    if event.stage.resource == resource
                )
                for resource in ("IB", "NVLink", "Compute")
            }
            variants.append(
                {
                    "policy": policy,
                    "staging_rows_per_edge": count * rows,
                    "wire_buffer_bytes_per_edge": count * rows * HIDDEN * 2,
                    "slots": count,
                    "completion": end,
                    "resource_busy_time": busy,
                }
            )
        best = min(row["completion"] for row in variants[1:])
        result[name.lower()] = {
            "variants": variants,
            "smallest_tested_slots_at_best_completion": min(
                row["slots"] for row in variants[1:] if row["completion"] == best
            ),
        }
    return result


def dependency_window(path, upstream_work):
    """Apply measured readiness events to an explicitly hypothetical consumer."""
    measured = json.loads(path.read_text())
    if measured["schema"] != 1 or measured["cc"] != [10, 3]:
        raise ValueError("expected the SM103 backward experiment report")
    cases = []
    for case in measured["cases"]:
        for work in upstream_work:
            completion = {}
            for policy, values in case["summary_us"].items():
                delivery = values["input_gradient_delivery_us"]["median"]
                backward = values["backward_completion_us"]["median"]
                completion[policy] = max(delivery + work, backward)
            cases.append(
                {
                    "rows": case["rows"],
                    "extra_live_bytes": case["extra_live_bytes"],
                    "assumed_upstream_work_us": work,
                    "completion_from_medians_us": completion,
                }
            )
    return {
        "evidence": str(path),
        "scope": "two-branch sensitivity calculation, not an observed training schedule",
        "formula": "max(input-gradient delivery + upstream work, backward completion)",
        "assumptions": [
            "The preceding stage is ready and runs independently after gradient delivery.",
            "No additional resource contention, communication or other dependencies.",
            "Maxima of medians are scenario inputs, not measured completion quantiles.",
        ],
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cp-gpus", type=int, default=4)
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--slots", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--samples", type=Path, nargs="+", default=[])
    parser.add_argument("--backward-report", type=Path)
    parser.add_argument("--upstream-work-us", type=float, nargs="+", default=[0, 32, 80])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if min(args.chunks, args.rows, *args.slots) < 1:
        parser.error("chunks, rows and slots must be positive")
    if any(not isfinite(t) or t < 0 for t in args.upstream_work_us):
        parser.error("upstream work must be finite and nonnegative")
    samples, probes, measured_rows = read_samples(args.samples)
    complete = probes == set(PROBES)
    if measured_rows is not None and measured_rows != args.rows:
        parser.error("--rows must match all supplied measurements")
    report = {
        "schema": 1,
        "context_layouts": context_layouts(args.cp_gpus),
        "expert_route": expert_route(),
        "pipeline": {
            "chunks": args.chunks,
            "rows_per_chunk": args.rows,
            "time_unit": "modeled microseconds" if complete else "dependency ticks",
            "missing_probes": sorted(set(PROBES) - probes),
            "samples": [str(path) for path in args.samples],
            "assumptions": [
                "One FIFO server each for IB, NVLink and compute; authored phase fences.",
                "Incomplete calibration uses one tick for every action, never mixed rates.",
                "Buffer bytes cover one BF16 wire edge, not all live training allocations.",
                "Isolated medians do not measure concurrent contention or training throughput.",
            ],
            "passes": route_schedules(
                args.chunks, sorted(set(args.slots)), args.rows, samples if complete else {}
            ),
        },
    }
    if args.backward_report:
        report["dependency_window"] = dependency_window(args.backward_report, args.upstream_work_us)
    rendered = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
