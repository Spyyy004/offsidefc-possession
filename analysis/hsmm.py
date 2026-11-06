"""Duration-aware decoder: hidden semi-Markov model (expert suggestion, Phase 2 #2).

States H, A, DH (dead, home had it), DA (dead, away had it). Each SEGMENT has an explicit length
scored by a duration prior learned from external labels; transitions between segments are learned
from the same labels. Decoding runs on 5-frame blocks (0.2 s) for speed, then every boundary is
refined to the exact frame with the per-frame evidence.

  score(segment of state s over blocks (u, t]) =
      gamma * sum of block log-evidence for s  +  lam * log P_dur_s(length)  +  log P(prev -> s)
"""
import numpy as np

BLOCK = 5
EPS = 1e-4
H, A, DH, DA = 0, 1, 2, 3


def frame_loglik(p_dead, p_home, beta):
    pd_ = np.clip(p_dead, EPS, 1 - EPS)
    ph = np.clip(p_home, EPS, 1 - EPS)
    alive = np.log1p(-pd_)
    ld = np.log(pd_)
    return np.stack([alive + beta * np.log(ph), alive + beta * np.log1p(-ph), ld, ld], axis=1)   # (n, 4)


def segments_of(label):
    lab = np.asarray(label)
    b = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    bounds = np.concatenate(([0], b, [len(lab)]))
    return [(lab[a], a, e) for a, e in zip(bounds[:-1], bounds[1:])]


def learn_priors(label_halves, max_blocks):
    """Duration log-pmfs (in blocks) for possession and dead segments, plus transition log-probs."""
    pos, dead = [], []
    trans = np.full((4, 4), 1.0)            # add-one counts
    for lab in label_halves:
        segs = segments_of(lab)
        for i, (l, a, e) in enumerate(segs):
            nb = max(1, int(round((e - a) / BLOCK)))
            (dead if l == "D" else pos).append(min(nb, max_blocks))
            if i + 1 < len(segs):
                nl = segs[i + 1][0]
                if l in "HA":
                    s = H if l == "H" else A
                    trans[s, (A if s == H else H) if nl in "HA" else (DH if s == H else DA)] += 1
                else:
                    prev = segs[i - 1][0] if i > 0 else None
                    s = DH if prev == "H" else DA
                    if nl in "HA":
                        trans[s, H if nl == "H" else A] += 1
    allowed = np.zeros((4, 4), bool)
    allowed[H, [A, DH]] = allowed[A, [H, DA]] = allowed[DH, [H, A]] = allowed[DA, [H, A]] = True
    trans = np.where(allowed, trans, 0)
    with np.errstate(divide="ignore"):
        log_t = np.where(allowed, np.log(trans / trans.sum(axis=1, keepdims=True)), -np.inf)

    def pmf(lengths):
        h = np.bincount(lengths, minlength=max_blocks + 1)[1:].astype(float) + 0.1
        k = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)                      # light smoothing
        h = np.convolve(h, k / k.sum(), mode="same")
        return np.log(h / h.sum())

    return pmf(np.array(pos)), pmf(np.array(dead)), log_t


def decode(p_dead, p_home, priors, beta=2.0, gamma=1.0, lam=1.0, refine=True):
    dur_pos, dur_dead, log_t = priors
    max_b = len(dur_pos)
    ll = frame_loglik(p_dead, p_home, beta)                                  # (n, 4)
    n = len(ll)
    T = int(np.ceil(n / BLOCK))
    pad = np.vstack([ll, np.repeat(ll[-1:], T * BLOCK - n, axis=0)])
    blk = pad.reshape(T, BLOCK, 4).sum(axis=1) * gamma                       # (T, 4)
    C = np.vstack([np.zeros((1, 4)), np.cumsum(blk, axis=0)])                # (T+1, 4)
    dur = np.stack([dur_pos, dur_pos, dur_dead, dur_dead]) * lam             # (4, max_b)

    delta = np.full((T + 1, 4), -np.inf)
    best_in = np.full((T + 1, 4), -np.inf)
    best_in[0] = 0.0                                                          # a segment may start at 0
    back_d = np.zeros((T + 1, 4), dtype=np.int32)
    back_s = np.full((T + 1, 4), -1, dtype=np.int8)
    from_state = np.full((T + 1, 4), -1, dtype=np.int8)
    for t in range(1, T + 1):
        dmax = min(max_b, t)
        d = np.arange(1, dmax + 1)
        u = t - d
        for s in range(4):
            cand = best_in[u, s] + dur[s, :dmax] + (C[t, s] - C[u, s])
            j = int(np.argmax(cand))
            delta[t, s] = cand[j]
            back_d[t, s] = d[j]
        # best way into each state for a segment starting after block t
        tot = delta[t][:, None] + log_t                                       # (from, to)
        fs = np.argmax(tot, axis=0)
        best_in[t] = tot[fs, np.arange(4)]
        from_state[t] = fs
    # backtrack
    s = int(np.argmax(delta[T]))
    t = T
    seg = []
    while t > 0:
        d = int(back_d[t, s])
        seg.append((t - d, t, s))
        u = t - d
        s = int(from_state[u, s]) if u > 0 else -1
        t = u
    seg.reverse()
    states = np.empty(T * BLOCK, dtype=np.int8)
    for a, e, s in seg:
        states[a * BLOCK:e * BLOCK] = s
    states = states[:n]
    if refine:
        states = refine_boundaries(states, ll)
    return np.array(["H", "A", "D", "D"])[states]


def refine_boundaries(states, ll, radius=BLOCK):
    """Move each boundary within +/- radius frames (never past a neighbouring boundary) to the split
    best explained by the two states: frames lo..k-1 take the left state, k..hi-1 the right one."""
    out = states.copy()
    b = np.flatnonzero(states[1:] != states[:-1]) + 1
    for j, i in enumerate(b):
        left, right = states[i - 1], states[i]
        prev_b = b[j - 1] if j > 0 else 0
        next_b = b[j + 1] if j + 1 < len(b) else len(states)
        lo = max(i - radius, prev_b + (next_b - prev_b > 2))       # keep at least a frame of the neighbour
        hi = min(i + radius, next_b - 1)
        if hi - lo < 1:
            continue
        diff = ll[lo:hi, left] - ll[lo:hi, right]
        cum = np.concatenate(([0.0], np.cumsum(diff)))               # score of each split k = lo..hi
        k = lo + int(np.argmax(cum))
        out[lo:k] = left
        out[k:hi] = right
    return out
