"""Render attention and expert ownership using computation names and explicit GPU owners."""

import argparse
from pathlib import Path

from visualize import SVG
from workload import CONTEXT, HIDDEN, TOP_K, cp_payload, expert_ids

OWNER = ("#246984", "#99542f", "#3a755b", "#705392")
PALE = ("#dcebf0", "#f4e5dc", "#dcece2", "#ece4f2")
MUTED, RULE, PANEL = "#465e69", "#ced8d5", "#eff3ef"
WIDTH = 1500


def ownership_grid(svg, x, y, by_head):
    cell = 58
    for index in range(4):
        svg.text(x + index * cell + 28, y - 25, "Heads", 12, 700, MUTED, "middle")
        svg.text(
            x + index * cell + 28,
            y - 8,
            f"{32 * index}–{32 * index + 31}",
            12,
            400,
            MUTED,
            "middle",
        )
        svg.text(x - 12, y + index * cell + 25, "Token", 13, 400, MUTED, "end")
        svg.text(x - 12, y + index * cell + 43, f"block {index + 1}", 13, 700, MUTED, "end")
        for head in range(4):
            owner = head if by_head else index
            svg.box(x + head * cell, y + index * cell, cell - 3, cell - 3, OWNER[owner], radius=4)
            svg.text(
                x + head * cell + 27,
                y + index * cell + 33,
                f"GPU {owner}",
                13,
                700,
                "white",
                "middle",
            )


def context(out):
    svg = SVG(
        WIDTH,
        1196,
        "Attention tensors and their GPU owners",
        "Ulysses exchanges token and head ownership. A compact ring keeps query blocks local and rotates the latent and positional-key blocks.",
    )
    svg.text(40, 51, "Attention tensors and their GPU owners", 34, 700)
    svg.text(
        42,
        82,
        f"{CONTEXT:,} tokens per sequence · 128 heads · four equal token blocks in the ownership diagrams",
        18,
        color=MUTED,
    )
    svg.box(34, 109, 714, 553, PANEL, radius=12)
    svg.box(764, 109, 702, 553, PANEL, radius=12)
    svg.text(54, 148, "Ulysses · move from token owners to head owners", 23, 700)
    svg.text(54, 179, "fwd  Project keys and values, then exchange queries / keys / values.", 16)
    svg.text(220, 224, "Before all-to-all", 18, 700, anchor="middle")
    svg.text(564, 224, "After all-to-all", 18, 700, anchor="middle")
    ownership_grid(svg, 117, 278, False)
    ownership_grid(svg, 459, 278, True)
    svg.arrow([(357, 392), (383, 392)], MUTED, width=3)
    svg.text(54, 550, "A cell names the GPU holding that token block and those heads.", 16, 700)
    svg.text(54, 581, "Each head owner attends over the full sequence.", 17)
    svg.text(54, 610, "fwd  Return the attention output to its token owner.", 16)
    svg.text(54, 638, "bwd  Reverse the exchanges and reduce-scatter positional-key gradients.", 16)

    svg.text(784, 148, "Compact ring · keep each query block local", 23, 700)
    svg.text(784, 179, "KV block = normalized latent + shared positional key.", 16)
    for step in range(4):
        svg.text(
            1010 + step * 119,
            248,
            "Local step" if step == 0 else f"Hop {step}",
            15,
            700,
            MUTED,
            "middle",
        )
    for rank in range(4):
        y = 268 + rank * 59
        svg.box(784, y, 160, 53, OWNER[rank], radius=5)
        svg.text(864, y + 22, f"Query block {rank + 1}", 16, 700, "white", "middle")
        svg.text(864, y + 42, f"stays on GPU {rank}", 13, 400, "white", "middle")
        for step in range(4):
            owner = (rank - step) % 4
            active = owner <= rank
            x = 955 + step * 119
            svg.box(x, y, 110, 53, PALE[owner] if active else "#e3e7e3", radius=5)
            svg.text(
                x + 55,
                y + 22,
                f"KV block {owner + 1}",
                14,
                700,
                OWNER[owner] if active else MUTED,
                "middle",
            )
            svg.text(
                x + 55,
                y + 42,
                "attend" if active else "causal skip",
                13,
                color=MUTED,
                anchor="middle",
            )
    svg.text(
        784, 550, "Future blocks travel onward while their attention math is skipped.", 16, 700
    )
    svg.text(784, 581, "fwd  Expand each arriving block into keys and values.", 16)
    svg.text(784, 610, "fwd  Merge softmax statistics and weighted value sums.", 16)
    svg.text(784, 638, "bwd  Replay blocks and return latent and positional-key gradients.", 16)

    svg.text(40, 714, "Communication per GPU, per attention layer, per sequence", 25, 700)
    svg.text(
        40,
        745,
        "Logical BF16 payload in GiB. All bars use the same scale. Gradient transfers also use BF16 here.",
        17,
        color=MUTED,
    )
    for x, label in [
        (49, "CP GPUs"),
        (226, "Ulysses fwd / bwd"),
        (592, "Ring fwd"),
        (858, "Ring bwd"),
        (1143, "KV expansions in fwd"),
    ]:
        svg.text(x, 789, label, 18, 700)
    svg.text(1143, 813, "Ulysses / Ring", 15, color=MUTED)
    degrees = (2, 4, 8, 16)
    totals = []
    for ranks in degrees:
        payload = cp_payload(ranks)
        split, ring = payload["Ulysses"], payload["Ring"]
        totals.append((sum(split) / 2**30, ring[0] / 2**30, sum(ring) / 2**30))
    maximum = max(max(row) for row in totals)
    for index, (ranks, values) in enumerate(zip(degrees, totals)):
        y = 835 + index * 39
        svg.box(40, y - 8, 1420, 1, RULE)
        svg.text(59, y + 19, ranks, 19, 700)
        for x, gib, color in zip((228, 594, 860), values, ("#366985", "#36806b", "#71578a")):
            bar = 205 * gib / maximum
            svg.box(x, y + 3, bar, 19, color, radius=2)
            svg.text(x + bar + 10, y + 20, f"{gib:.2f}", 18, 700, color)
        svg.text(1150, y + 19, f"{ranks}  /  {ranks * (ranks + 1) // 2}", 18, 700)
    svg.text(
        40,
        1013,
        "Ulysses sends the displayed amount in each pass. KV expansions count whole token-block projections across the CP group.",
        16,
    )
    svg.text(
        40,
        1041,
        "The ring bwd total includes KV replay, accumulated gradients and the final gradient-only hop to the original owner.",
        16,
        color=MUTED,
    )
    svg.box(34, 1066, 1432, 105, PANEL, radius=10)
    same = cp_payload(4)["Ring"][0] / 2**30
    svg.text(
        54,
        1099,
        f"Compact all-gather control · four GPUs · {same:.2f} GiB fwd and {same:.2f} GiB bwd",
        21,
        700,
    )
    svg.text(
        54,
        1128,
        "Each query owner retains all four latent / positional-key blocks and reduce-scatters their gradients.",
        17,
    )
    svg.text(
        54,
        1154,
        "The ring consumes successive blocks and replays them in bwd. These figures compare payload and residency.",
        16,
        color=MUTED,
    )
    svg.save(out)


def row_packets(svg, x, y, owners, compact=False):
    groups = list(dict.fromkeys(owners)) if compact else owners
    for index, owner in enumerate(groups):
        width = 82 if compact else 49
        left = x + index * (width + 5)
        svg.box(left, y, width, 35, OWNER[owner], radius=4)
        svg.text(
            left + width / 2,
            y + 23,
            f"GPU {owner}",
            12 if not compact else 15,
            700,
            "white",
            "middle",
        )


def experts(out):
    choice = expert_ids(0, "colocated")
    owners = [expert // TOP_K for expert in choice]
    unique = tuple(dict.fromkeys(owners))
    row_kib = HIDDEN * 2 // 1024
    svg = SVG(
        WIDTH,
        1084,
        "Follow one token through expert dispatch and combine",
        "Eight selected experts on two destination GPUs. Rows represent token activations and expert outputs, with explicit forward and backward reductions.",
    )
    svg.text(40, 51, "Follow one token through expert dispatch and combine", 33, 700)
    svg.text(
        42,
        82,
        f"Top {TOP_K} selected experts · {HIDDEN:,} channels · one BF16 token activation row = {row_kib} KiB",
        18,
        color=MUTED,
    )
    svg.box(34, 111, 300, 195, PALE[0], radius=12)
    svg.text(54, 146, "Source GPU 0", 24, 700, OWNER[0])
    svg.text(54, 175, "Node 0 · owns the input token", 16)
    svg.text(54, 213, "token activation", 19, 700)
    for column in range(8):
        svg.box(55 + column * 29, 230, 25, 30, OWNER[0], radius=3)
    svg.text(54, 284, "One row before expert dispatch", 15, color=MUTED)
    for owner, label, x in [
        (1, "Node 0 · same-node experts", 405),
        (2, "Node 1 · remote experts", 950),
    ]:
        svg.box(x, 111, 516, 195, PALE[owner], radius=12)
        svg.text(x + 20, 146, f"Expert GPU {owner}", 24, 700, OWNER[owner])
        svg.text(x + 20, 175, label, 16)
        selected = [expert for expert in choice if expert // TOP_K == owner]
        for index, expert in enumerate(selected):
            left = x + 20 + index * 121
            svg.box(left, 201, 111, 52, OWNER[owner], radius=5)
            svg.text(left + 55, 233, f"Expert {expert}", 16, 700, "white", "middle")
        svg.text(x + 20, 284, "Four gate/up → SwiGLU → down computations", 16, color=MUTED)
    svg.arrow([(339, 210), (395, 210)], OWNER[1])
    svg.arrow([(362, 210), (362, 100), (932, 100), (932, 210), (946, 210)], OWNER[2])
    svg.text(
        42,
        342,
        "Each box below is one activation or output row. Its label identifies the destination or contributing GPU.",
        18,
        color=MUTED,
    )
    for x, label in [
        (43, "Protocol"),
        (262, "fwd · token activation dispatch"),
        (750, "fwd · expert output combine"),
        (1240, "Payload / direction"),
    ]:
        svg.text(x, 385, label, 17, 700)
    for y, title, sub in [
        (420, "Naive EP", "Per-assignment exchange"),
        (548, "Compact DeepEP", "Per-destination exchange"),
        (676, "Dist-MoE style", "Peer-addressed expert tiles"),
    ]:
        svg.box(40, y - 15, 1420, 1, RULE)
        svg.text(44, y + 25, title, 21, 700)
        svg.text(44, y + 52, sub, 14, color=MUTED)
    row_packets(svg, 262, 421, owners)
    row_packets(svg, 750, 421, owners)
    svg.text(262, 479, "One activation row for each selected expert.", 16)
    svg.text(750, 479, "Source GPU 0 weights and sums eight outputs.", 16)
    svg.text(1240, 447, f"{TOP_K * row_kib} KiB", 23, 700)
    svg.text(1240, 478, "8 activation rows", 16, color=MUTED)
    row_packets(svg, 262, 549, owners, True)
    row_packets(svg, 750, 549, owners, True)
    svg.text(262, 607, "Two rows fan out locally to eight expert computations.", 16)
    svg.text(750, 607, "Each expert GPU weights and combines its outputs.", 16)
    svg.text(1240, 575, f"{len(unique) * row_kib} KiB", 23, 700)
    svg.text(1240, 606, "2 activation rows", 16, color=MUTED)
    svg.box(262, 676, 250, 38, PALE[0], radius=5)
    svg.text(387, 701, "Source GPU 0 publishes the row", 15, 700, OWNER[0], "middle")
    svg.text(262, 735, "Gate/up projection gathers each tile by source address.", 16)
    svg.text(750, 700, "Down projection writes to each original token slot.", 16, 700)
    svg.text(750, 735, "Source GPU 0 weights and sums the completed outputs.", 16)
    svg.text(1240, 700, "8 addressed reads", 18, 700)
    svg.text(1240, 733, "8 output writes", 17, color=MUTED)
    svg.box(34, 779, 1432, 222, PANEL, radius=12)
    svg.text(54, 816, "bwd · output gradients travel out and input gradients return", 24, 700)
    svg.text(54, 854, "Naive / peer", 18, 700)
    svg.text(
        233,
        854,
        "Source GPU 0 weights an output gradient per expert slot, then sums the returning input gradients.",
        17,
    )
    svg.text(54, 892, "Compact", 18, 700)
    svg.text(
        233,
        892,
        "Each destination receives one output gradient, weights its expert slots and sums their input gradients locally.",
        17,
    )
    svg.text(233, 924, "Selected-score gradients also return to the source GPU.", 17)
    svg.text(
        54,
        970,
        "Expert owners accumulate the gate/up and down weight gradients in every plan.",
        18,
        700,
        MUTED,
    )
    svg.text(
        40,
        1040,
        "Payloads count activation data in one direction. Scores, indices and protocol overhead need separate accounting.",
        16,
        color=MUTED,
    )
    svg.text(
        40,
        1064,
        "Peer reads and writes count logical accesses. Their physical traffic depends on the implementation and fabric.",
        16,
        color=MUTED,
    )
    svg.save(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    context(args.out_dir / "context-parallel.svg")
    experts(args.out_dir / "expert-parallel.svg")
