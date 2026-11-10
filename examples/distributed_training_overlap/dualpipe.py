"""Check a small bidirectional PP schedule, split backward and activation-slot reuse."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch
from dualpipe_schedule import actions
from model import EXPERT, HIDDEN, ROWS, TrainingReference

from tilefoundry.analysis import BufferAccess, BufferLifetime, check_buffer_lifetimes


def dependency(action, ranks):
    kind, direction, batch, stage = action
    if kind == "fwd":
        return [] if stage == 0 else [(kind, direction, batch, stage - 1)]
    if kind == "bwd_weight":
        return [("bwd_input", direction, batch, stage)]
    return [("fwd", direction, batch, stage)] + (
        [] if stage == ranks - 1 else [("bwd_input", direction, batch, stage + 1)]
    )


def schedule(ranks, microbatches):
    groups = actions(ranks, microbatches)
    cursor, available = [0] * ranks, [0.0] * ranks
    done, arrived, records, links = {}, {}, [], {}
    while any(cursor[r] < len(groups[r]) for r in range(ranks)):
        choices = []
        for rank in range(ranks):
            if cursor[rank] == len(groups[rank]):
                continue
            group = groups[rank][cursor[rank]]
            external = {d for a in group for d in dependency(a, ranks)} - set(group)
            if external <= done.keys():
                readiness = [
                    (arrived if d[3] != a[3] else done)[d]
                    for a in group
                    for d in dependency(a, ranks)
                    if d in external
                ]
                choices.append((max([available[rank], *readiness]), rank))
        if not choices:
            raise ValueError("pipeline dependencies cannot make progress")
        time, rank = min(choices)
        for action in groups[rank][cursor[rank]]:
            kind, direction, batch, stage = action
            duration = {"fwd": 3, "bwd_input": 2, "bwd_weight": 1}[kind]
            end = time + duration
            records.append(
                {
                    "action": action,
                    "rank": rank,
                    "group": cursor[rank],
                    "start": time,
                    "end": end,
                    "resource": f"gpu{rank}.compute",
                }
            )
            done[action] = end
            arrived[action] = end
            if (kind == "fwd" and stage < ranks - 1) or (kind == "bwd_input" and stage > 0):
                neighbor = rank + (1 if direction == 0 else -1) * (1 if kind == "fwd" else -1)
                link = f"gpu{rank}->gpu{neighbor}"
                start = max(end, links.get(link, 0))
                links[link] = start + 1
                records.append(
                    {
                        "action": ("send_" + kind, direction, batch, stage),
                        "rank": rank,
                        "group": cursor[rank],
                        "start": start,
                        "end": start + 1,
                        "resource": link,
                    }
                )
                arrived[action] = start + 1
            time = end
        available[rank] = time
        cursor[rank] += 1
    return sorted(records, key=lambda event: (event["start"], event["end"], event["rank"]))


def activation_slots(records):
    events = {tuple(e["action"]): e for e in records if not e["action"][0].startswith("send_")}
    # Per-stage saved state: one input plus gate, up and hidden for five experts.
    nbytes = ROWS * (HIDDEN + 5 * 3 * EXPERT) * 4
    free, values, assignments = {}, [], []
    for event in records:
        kind, direction, batch, stage = event["action"]
        if kind != "fwd":
            continue
        rank = event["rank"]
        consumers = [events[k, direction, batch, stage] for k in ("bwd_input", "bwd_weight")]
        ends = free.setdefault(rank, [])
        slot = next((i for i, end in enumerate(ends) if end <= event["start"]), len(ends))
        release = max(e["end"] for e in consumers)
        if slot == len(ends):
            ends.append(release)
        else:
            ends[slot] = release
        name = f"stage{stage}.direction{direction}.microbatch{batch}"
        value = BufferLifetime(
            name,
            f"gpu{rank}.activation",
            slot * nbytes,
            nbytes,
            BufferAccess(name + ".fwd", event["start"], event["end"]),
            tuple(
                BufferAccess(name + "." + e["action"][0], e["start"], e["end"]) for e in consumers
            ),
        )
        values.append(value)
        assignments.append({"value": name, "rank": rank, "slot": slot, "release": release})
    assert not check_buffer_lifetimes(tuple(values))
    # Negative case: reuse immediately after input backward while weight backward still owns state.
    old = next(v for v in values if v.consumers[0].end < v.consumers[1].end)
    time = old.consumers[0].end
    bad = replace(
        old,
        value="premature_next_forward",
        producer=BufferAccess("premature_forward", time, time + 0.1),
        consumers=(),
    )
    hazards = check_buffer_lifetimes((old, bad))
    assert hazards and hazards[0].kind == "overwrite_live_value"
    return {
        "slots_per_rank": {str(r): len(ends) for r, ends in free.items()},
        "bytes_per_slot": nbytes,
        "assignments": assignments,
        "negative_case": [asdict(h) for h in hazards],
    }


def check(ranks=4, microbatches=8):
    records = schedule(ranks, microbatches)
    reference = TrainingReference(ranks, microbatches)
    compute = [e for e in records if not e["action"][0].startswith("send_")]
    assert len({tuple(e["action"]) for e in compute}) == 3 * ranks * microbatches
    # Completion order is a legal sequential replay of the modeled operations.
    for event in sorted(compute, key=lambda e: e["end"]):
        reference.execute(event["action"])
    error = reference.validate()
    return {
        "ranks": ranks,
        "microbatches": microbatches,
        "time_unit": "dependency ticks",
        "paired_group_policy": "serialize compute within each upstream overlap opportunity",
        "max_abs_error": error,
        "events": records,
        "activation_slots": activation_slots(records),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--microbatches", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("results/dualpipe.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = check(args.ranks, args.microbatches)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"{args.ranks} PP ranks, {args.microbatches} microbatches: max error {report['max_abs_error']:.3g}; "
        f"slots {report['activation_slots']['slots_per_rank']}; premature reuse rejected"
    )
