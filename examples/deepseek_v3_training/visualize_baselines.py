"""Render the two ownership comparisons using the example's shapes and controlled routes."""

import argparse
from pathlib import Path

from visualize import SVG
from workload import CONTEXT, HIDDEN, TOP_K, cp_payload, expert_ids

RANK = ("#246984", "#99542f", "#3a755b", "#705392")
PALE = ("#dcebf0", "#f4e5dc", "#dcece2", "#ece4f2")
MUTED, RULE = "#465e69", "#ced8d5"
W = 1240


def panel(svg, x, y, w, h):
    svg.box(x, y, w, h, "#f0f3ef")


def rank_grid(svg, x, y, by_head):
    cell = 26
    for i in range(4):
        svg.text(x + i * cell + 12, y - 7, f"H{i}", 12, 700, MUTED, "middle")
        svg.text(x - 8, y + i * cell + 18, f"T{i}", 12, 700, MUTED, "end")
        for j in range(4):
            rank = j if by_head else i
            svg.box(x + j * cell, y + i * cell, 24, 24, RANK[rank])
            svg.text(x + j * cell + 12, y + i * cell + 16, rank, 12, 700, "white", "middle")


def context(out):
    s = SVG(
        W,
        879,
        "Two ways to distribute causal MLA",
        "Ownership grids for Ulysses and a compact latent ring, followed by per-rank forward and backward payloads.",
    )
    s.text(40, 52, "Two ways to distribute causal MLA", 34, 700)
    s.text(
        42,
        82,
        f"{CONTEXT:,} tokens in one sequence   ·   128 attention heads   ·   four ranks in the ownership diagrams",
        17,
        400,
        MUTED,
    )
    panel(s, 34, 105, 568, 334)
    panel(s, 618, 105, 588, 334)
    s.text(53, 139, "Ulysses", 24, 700)
    s.text(53, 164, "A token owner projects K/V before the head exchange.", 16)
    s.text(92, 198, "Token shards", 15, 700)
    s.text(361, 198, "Head shards", 15, 700)
    rank_grid(s, 97, 226, False)
    rank_grid(s, 368, 226, True)
    s.text(276, 267, "all-to-all", 15, 700, MUTED, "middle")
    s.text(276, 302, "→", 32, 700, MUTED, "middle")
    s.text(53, 361, "Cell = rank owning (token block, head group).", 16, 700)
    s.text(53, 388, "Gather the shared position key. Send Y back to token owners.", 15)
    s.text(53, 414, "Backward reverses both transposes and reduce-scatters dKey.", 15)
    s.text(638, 139, "Compact ring", 24, 700)
    s.text(638, 164, "Each query owner keeps Q and visits successive KV blocks.", 16)
    for step in range(4):
        s.text(785 + step * 92, 205, f"hop {step}", 14, 700, MUTED, "middle")
    for rank in range(4):
        y = 214 + rank * 29
        s.box(639, y, 76, 24, RANK[rank])
        s.text(677, y + 17, f"Q{rank}", 14, 700, "white", "middle")
        for step in range(4):
            owner = (rank - step) % 4
            active = owner <= rank
            x = 748 + step * 92
            s.box(x, y, 74, 24, PALE[owner] if active else "#e4e7e4")
            s.text(
                x + 37,
                y + 17,
                f"KV{owner}" if active else f"KV{owner} ×",
                14,
                700,
                RANK[owner] if active else "#646f72",
                "middle",
            )
    s.text(638, 361, "×  Future block travels onward. Its math is skipped.", 16, 700)
    s.text(638, 388, "Merge row maxima, normalizers and value sums as blocks arrive.", 15)
    s.text(638, 414, "Backward replays KV and returns each summed dKV to its owner.", 15)
    s.text(40, 487, "Logical payload sent per rank", 24, 700)
    s.text(
        40,
        516,
        "GiB per sequence per MLA layer, BF16 payloads and adjoints. Bar lengths share one scale.",
        16,
        400,
        MUTED,
    )
    degrees = (2, 4, 8, 16)
    totals = []
    for p in degrees:
        payload = cp_payload(p)
        split, ring = payload["Ulysses"], payload["Ring"]
        totals.append((sum(split) / 2**30, ring[0] / 2**30, sum(ring) / 2**30))
    for x, label in [
        (48, "CP ranks"),
        (188, "Ulysses · F or B"),
        (503, "Ring · F"),
        (762, "Ring · B"),
        (984, "KV projections¹"),
    ]:
        s.text(x, 558, label, 16, 700)
    maximum = max(max(row) for row in totals)
    for i, (p, numbers) in enumerate(zip(degrees, totals)):
        y = 575 + i * 35
        s.box(40, y - 3, 1160, 1, RULE)
        s.text(54, y + 19, p, 17, 700)
        for x, gib, color in zip((190, 505, 764), numbers, ("#366985", "#36806b", "#71578a")):
            bar = 150 * gib / maximum
            s.box(x, y + 4, bar, 14, color)
            s.text(x + bar + 8, y + 17, f"{gib:.2f}", 15, 700, color)
        s.text(992, y + 18, f"{p}  /  {p * (p + 1) // 2}", 16, 700)
    s.text(
        40,
        747,
        "¹ Ulysses / ring, across one CP group in forward. One projection expands one token block for all heads.",
        15,
    )
    s.text(
        40,
        777,
        "The ring sends 512 latent + 64 positional values per token. Backward includes the final gradient-only hop home.",
        15,
        400,
        MUTED,
    )
    panel(s, 34, 799, 1172, 62)
    same = cp_payload(4)["Ring"][0] / 2**30
    s.text(
        52,
        824,
        f"Same compact KV, all-gather control at CP 4: {same:.2f} GiB forward and {same:.2f} GiB backward.",
        17,
        700,
    )
    s.text(
        52,
        847,
        "Each query rank retains all four blocks and reduce-scatters their gradients. The ring consumes successive blocks and replays them in backward.",
        15,
    )
    s.save(out)


def strips(s, x, y, owners, compact=False):
    left = x
    groups = list(dict.fromkeys(owners)) if compact else owners
    for owner in groups:
        width = 56 if compact else 23
        s.box(left, y, width, 22, RANK[owner])
        s.text(left + width / 2, y + 16, f"r{owner}", 12, 700, "white", "middle")
        left += width + 3


def experts(out):
    choice = expert_ids(0, "colocated")
    owners = [e // TOP_K for e in choice]
    unique = tuple(dict.fromkeys(owners))
    kib = HIDDEN * 2 // 1024
    s = SVG(
        W,
        838,
        "One token through three expert dispatchers",
        "Eight selected experts share two destination ranks. The diagram counts logical activation rows and shows where the backward reductions run.",
    )
    s.text(40, 52, "One token through three expert dispatchers", 33, 700)
    s.text(
        42,
        82,
        f"Token owner r0   ·   top {TOP_K}   ·   {HIDDEN:,} channels   ·   one BF16 activation row = {kib} KiB",
        17,
        400,
        MUTED,
    )
    panel(s, 34, 107, 1172, 141)
    s.text(54, 138, "Selected experts for token 0", 20, 700)
    for i, (owner, e) in enumerate(zip(owners, choice)):
        x = 55 + i * 65
        s.box(x, 158, 58, 33, RANK[owner])
        s.text(x + 29, 180, f"e{e}", 15, 700, "white", "middle")
    s.text(54, 218, "r1  ·  same node", 15, 700, RANK[1])
    s.text(315, 218, "r2  ·  other node", 15, 700, RANK[2])
    s.text(641, 170, "Eight expert MLPs run in every plan.", 20, 700)
    s.text(641, 196, "The plans differ in which rows cross an ownership boundary", 16)
    s.text(641, 219, "and where the token’s weighted contributions are reduced.", 16)
    for x, words in [
        (45, "Protocol"),
        (252, "Activation dispatch"),
        (570, "Expert output and combine"),
        (960, "Activation payload¹"),
    ]:
        s.text(x, 290, words, 16, 700, MUTED)
    rows = [
        (313, "Naive EP", "Variable-split exchange"),
        (422, "Compact DeepEP", "Deduplicated layout"),
        (531, "Dist-MoE style", "Peer-addressed GEMMs"),
    ]
    for y, title, sub in rows:
        s.box(39, y - 8, 1162, 1, RULE)
        s.text(45, y + 27, title, 19, 700)
        s.text(45, y + 54, sub, 14, 400, MUTED)
    strips(s, 252, 323, owners)
    s.text(252, 369, "One row per selected expert.", 15)
    strips(s, 570, 323, owners)
    s.text(570, 369, "Return eight Y rows. r0 weights and sums.", 15)
    s.text(960, 345, f"{TOP_K} × {kib} = {TOP_K * kib} KiB", 18, 700)
    strips(s, 252, 432, owners, True)
    s.text(252, 478, "Two rows fan out to eight experts.", 15)
    strips(s, 570, 432, owners, True)
    s.text(570, 478, "Weight and sum locally. r0 adds two rows.", 15)
    s.text(960, 454, f"{len(unique)} × {kib} = {len(unique) * kib} KiB", 18, 700)
    s.box(252, 540, 121, 26, PALE[0])
    s.text(312, 558, "r0 publishes X", 14, 700, RANK[0], "middle")
    s.text(252, 588, "W13 gathers by (rank, row).", 15)
    s.text(570, 555, "W2 writes to eight origin slots.", 16, 700)
    s.text(570, 588, "r0 weights and sums after peer writes finish.", 15)
    s.text(960, 554, "8 addressed reads", 17, 700)
    s.text(960, 581, "8 route writes", 16)
    panel(s, 34, 634, 1172, 131)
    s.text(54, 665, "Read the routes backward", 20, 700)
    s.text(
        54,
        695,
        "Naive and peer-addressed paths weight dY per slot at r0. The token owner sums returning dX contributions.",
        16,
    )
    s.text(
        54,
        722,
        "Compact dispatch sends dY once per owner. Each owner weights expert rows, adds their dX locally and returns selected-score gradients.",
        16,
    )
    s.text(
        54,
        748,
        "Every plan accumulates W13 and W2 gradients at the expert and produces one dX per source token.",
        15,
        700,
        MUTED,
    )
    s.text(
        40,
        805,
        "¹ Payload for one direction of forward in this route. Counts exclude scores, indices and protocol overhead. Peer addresses count logical accesses.",
        15,
        400,
        MUTED,
    )
    s.save(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("."))
    args = p.parse_args()
    context(args.out_dir / "context-parallel.svg")
    experts(args.out_dir / "expert-parallel.svg")
