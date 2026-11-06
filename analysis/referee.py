"""Layer 3: the match-flow referee -- a 4-state HMM decoded with Viterbi (rules R2, R3, R4).

States: H, A, DH (dead, home had it before), DA (dead, away had it before).
Evidence per frame (log-probabilities) comes from Layer 2:
    H  : log(1 - p_dead) + beta * log(p_home)
    A  : log(1 - p_dead) + beta * log(1 - p_home)
    DH, DA : log(p_dead)
Costs for changing state (subtracted, in the same log units):
    switch        H <-> A          change of possession in open play (R4)
    stop          H -> DH, A -> DA play stops (R2)
    restart_other DH -> A, DA -> H restart goes to the other team (R3, the usual case)
    restart_same  DH -> H, DA -> A same team restarts
Any other move (e.g. H -> DA) is impossible.

Viterbi finds the single most believable state sequence for the whole half.
"""
import numpy as np

H, A, DH, DA = 0, 1, 2, 3
EPS = 1e-4

# Chosen in Step 8b by nested leave-one-match-out on the Bundesliga bench (D27): picked by all 7
# folds in round 2; round 3 confirmed a plateau. Order: switch, stop, restart_other, restart_same, beta.
REFEREE_SETTINGS = dict(switch=10, stop=160, restart_other=5, restart_same=5, beta=2.0)


def viterbi(p_dead, p_home, switch, stop, restart_other, restart_same, beta=1.0):
    """Most likely H/A/D label per frame."""
    pd_ = np.clip(p_dead, EPS, 1 - EPS)
    ph = np.clip(p_home, EPS, 1 - EPS)
    alive = np.log1p(-pd_)
    eh = (alive + beta * np.log(ph)).tolist()
    ea = (alive + beta * np.log1p(-ph)).tolist()
    ed = np.log(pd_).tolist()
    n = len(eh)
    # switch may be one number or a per-frame array ("change moments": cheap where a turnover is likely)
    sw = (np.full(n, float(switch)) if np.isscalar(switch) else np.asarray(switch, dtype=float)).tolist()
    back = np.zeros((n, 4), dtype=np.int8)
    # start: alive states free, dead states free (a half may start dead)
    sh, sa, sdh, sda = eh[0], ea[0], ed[0], ed[0]
    for t in range(1, n):
        # into H: from H (stay), A (switch), DH (restart_same), DA (restart_other)
        s_t = sw[t]
        c = (sh, sa - s_t, sdh - restart_same, sda - restart_other)
        bh = max(range(4), key=c.__getitem__); nh = c[bh] + eh[t]
        c2 = (sh - s_t, sa, sdh - restart_other, sda - restart_same)
        ba = max(range(4), key=c2.__getitem__); na = c2[ba] + ea[t]
        # into DH: from H (stop) or DH (stay); into DA: from A (stop) or DA (stay)
        if sh - stop > sdh:
            bdh, ndh = H, sh - stop + ed[t]
        else:
            bdh, ndh = DH, sdh + ed[t]
        if sa - stop > sda:
            bda, nda = A, sa - stop + ed[t]
        else:
            bda, nda = DA, sda + ed[t]
        back[t, 0], back[t, 1], back[t, 2], back[t, 3] = bh, ba, bdh, bda
        sh, sa, sdh, sda = nh, na, ndh, nda
    state = int(np.argmax([sh, sa, sdh, sda]))
    path = np.empty(n, dtype=np.int8)
    for t in range(n - 1, -1, -1):
        path[t] = state
        state = back[t, state]
    return np.array(["H", "A", "D", "D"])[path]
