"""
Generates src/shared/gpurtMergePackLut.h: the min-node packing LUT used by
PackLeafOnlyMergeGroups() (MergeScratchNodes.hlsli, WideNodeMergingFlags 0x10).

Based on /home/grvermeer/pack8 (gen_table.py, pack8.cuh).

Problem: a parent has up to 8 box children. Child size = numBoxes + numLeaves (1..8).
Merging packs consecutive children into one node while the size sum stays <= 8.
For every multiset of 8 sizes (C(15,8) = 6435) the table stores the order of the
sorted sizes that gives the fewest packed nodes with the consecutive greedy.

Entry layout (uint32):
  bits  0..23 : sorted index for table slot t = (e >> 3*t) & 7
  bits 24..31 : bit (24+t) = 1 if table slot t starts a new packed node
In every entry the 1..7 groups come first and all 8s last (ascending sorted index).
Non-box lanes (the tail) are treated as virtual 8s, so the same table covers
parents with fewer than 8 box children.

Also verifies, lane by lane, the exact algorithm the shader runs
(counting sort instead of the bitonic sort in pack8.cuh, "apply only when the
node count drops" rule).

Usage: python3 genMergePackLut.py [output header path]
"""
from itertools import combinations_with_replacement, permutations
from math import comb
import os
import random
import sys

C = 8  # capacity of one packed node
N = 8  # lanes (box child slots)

ENTRY_COUNT = comb(15, 8)          # 6435
UINT4_COUNT = (ENTRY_COUNT + 3) // 4  # 1609


def rank_sorted(v):  # v: ascending, values 1..8
    return sum(comb((v[i] - 1) + i, i + 1) for i in range(N))


def solve(a):  # exact bitmask DP, returns (min nodes, index order)
    n = len(a)
    INF = (99, 99)
    dp = [INF] * (1 << n)
    prev = [None] * (1 << n)
    dp[0] = (1, 0)
    for m in range(1 << n):
        if dp[m] == INF:
            continue
        b, s = dp[m]
        for i in range(n):
            if m >> i & 1:
                continue
            nxt = (b, s + a[i]) if s + a[i] <= C else (b + 1, a[i])
            if nxt < dp[m | 1 << i]:
                dp[m | 1 << i] = nxt
                prev[m | 1 << i] = (m, i)
    order, m = [], (1 << n) - 1
    while m:
        m, i = prev[m]
        order.append(i)
    return (dp[-1][0] if n else 0), order[::-1]


def block_starts(vals):
    mask, s = 0, C + 1
    for j, x in enumerate(vals):
        if s + x > C:
            mask |= 1 << j
            s = x
        else:
            s += x
    return mask


def greedy_count(vals):
    return bin(block_starts(vals)).count("1")


opt_cache = {}


def opt_nodes(ms):  # brute force over all orders (independent of the DP)
    key = tuple(sorted(ms))
    if key not in opt_cache:
        opt_cache[key] = min(greedy_count(p) for p in set(permutations(key))) if key else 0
    return opt_cache[key]


# ---------- build table (same as pack8/gen_table.py) ----------
table = [None] * ENTRY_COUNT
for ms in combinations_with_replacement(range(1, 9), N):
    nb, order = solve(ms)
    L = sum(1 for x in ms if x != 8)  # sorted: 8s occupy indices L..7
    order = [i for i in order if ms[i] != 8] + list(range(L, N))
    vals = [ms[i] for i in order]
    starts = block_starts(vals)
    assert bin(starts).count("1") == nb
    e = 0
    for t, i in enumerate(order):
        e |= i << (3 * t)
    e |= starts << 24
    r = rank_sorted(ms)
    assert table[r] is None
    table[r] = e
assert all(t is not None for t in table)


# ---------- lane-level simulation of PackLeafOnlyMergeGroups() ----------
def shader_sim(x):
    """x[l]: child size of box slot l (1..8), or 0 for a non-box slot (must be a tail).
    Returns (srcLane per slot, group start mask); identity and 0 when nothing changes."""
    lane = range(N)
    identity = (list(lane), 0)
    is_box = [x[l] > 0 for l in lane]
    box_mask = sum(1 << l for l in lane if is_box[l])
    k = bin(box_mask).count("1")
    if box_mask != (1 << k) - 1:  # box slots must be a prefix
        return identity
    v = [x[l] if is_box[l] else 8 for l in lane]  # non-box -> virtual 8
    cand_mask = sum(1 << l for l in lane if is_box[l] and v[l] < 8)
    if bin(cand_mask).count("1") < 2:
        return identity

    # baseline: consecutive greedy on the current order (same as MergeLeafOnlyBoxNodes)
    base = greedy_count([v[l] for l in lane if is_box[l]])

    # counting sort: pos = number of smaller keys
    key = [((v[l] - 1) << 3) | l for l in lane]
    pos = [sum(1 for j in lane if key[j] < key[l]) for l in lane]

    # rank term C(sv + pos, pos + 1), computed like the shader (exact integer steps)
    term = []
    for l in lane:
        sv, p, t = v[l] - 1, pos[l], 1
        for i in range(1, 9):
            if i <= p + 1:
                t = t * (sv - 1 + i) // i
        term.append(t)
    rank = sum(term)
    e = table[rank]

    c8 = sum(1 for l in lane if v[l] == 8)
    L = N - c8
    real8 = c8 - (N - k)

    def slot(p):
        if p < real8:
            return L + p
        if p < k:
            return p - real8
        return p

    t = [slot(p) for p in lane]
    src_sorted = [(e >> (3 * t[p])) & 7 for p in lane]
    starts = sum(1 << p for p in lane if p < k and (e >> (24 + t[p])) & 1)
    if bin(starts).count("1") >= base:
        return identity
    src_lane = []
    for p in lane:
        s = p
        for j in lane:
            if pos[j] == src_sorted[p]:
                s = j
        src_lane.append(s)
    return src_lane, starts


def merge_sim(vals, forced):  # MergeLeafOnlyBoxNodes with forcedGroupStartMask (capacity only)
    count, size = 0, 0
    for j, x in enumerate(vals):
        if count == 0 or ((forced >> j) & 1) or size + x > C:
            count += 1
            size = x
        else:
            size += x
    return count


random.seed(1)
checked = 0
changed = 0
for k in range(N + 1):
    for ms in combinations_with_replacement(range(1, 9), k):
        perms = [list(ms)] + [random.sample(ms, k) for _ in range(20)]
        for vals in perms:
            x = vals + [0] * (N - k)
            src, starts = shader_sim(x)
            assert sorted(src) == list(range(N))                  # a permutation
            assert src[k:] == list(range(k, N))                    # non-box tail untouched
            out = [x[s] for s in src]
            base = greedy_count(vals)
            got = merge_sim(out[:k], starts)
            if starts == 0:
                assert src == list(range(N))
                assert got == base
                # unchanged only when the current order is already optimal
                assert base == opt_nodes(vals) or sum(1 for a in vals if a < 8) < 2
            else:
                changed += 1
                assert got == bin(starts).count("1") == opt_nodes(vals) < base
                assert starts & ~((1 << k) - 1) == 0
                size = 0                                           # every group fits

                for j in range(k):
                    size = out[j] if (starts >> j) & 1 else size + out[j]
                    assert size <= C
            checked += 1
print("verified", checked, "inputs;", changed, "would be reordered; entries:", ENTRY_COUNT)

# ---------- emit header ----------
out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src", "shared", "gpurtMergePackLut.h")
padded = table + [0] * (UINT4_COUNT * 4 - ENTRY_COUNT)
with open(out_path, "w") as f:
    f.write("// Generated by tools/genMergePackLut.py. Do not edit.\n")
    f.write("// Min-node packing LUT for leaf-only box merging (WideNodeMergingFlags 0x10).\n")
    f.write("// %d entries = all multisets of 8 child sizes in 1..8, indexed by the combinatorial rank of the\n" % ENTRY_COUNT)
    f.write("// sorted sizes (rank = sum_l C(sv_l + l, l + 1), sv_l = l-th smallest size - 1). Padded to %d uint4.\n" % UINT4_COUNT)
    f.write("//   bits  0..23: sorted index for table slot t = (e >> 3*t) & 7\n")
    f.write("//   bits 24..31: bit (24+t) set if table slot t starts a new packed node\n")
    f.write("// In each entry the 1..7 groups come first and all 8s last (ascending).\n")
    f.write("#pragma once\n\n")
    f.write("#define GPURT_MERGE_PACK_LUT_DATA \\\n")
    lines = []
    for i in range(0, len(padded), 8):
        lines.append("    " + ", ".join("0x%08Xu" % t for t in padded[i:i + 8]))
    f.write(", \\\n".join(lines))
    f.write("\n")
print("wrote", os.path.normpath(out_path))
