import numpy as np
from envs.battleship_rl_env import BOARD_SIZE, FLEET

# status w obs 2D: 0 unknown, 1 miss, 2 hit(żywy lub ogólnie hit)

def _fits(cells, obs):
    # statek pasuje jeśli:
    # - żadna komórka nie jest miss
    # - wszystkie hit'y na planszy, które są w linii tego statku, są pokryte
    for r, c in cells:
        if obs[r, c] == 1:  # miss
            return False
    return True


def build_probability_heatmap(obs, remaining_lengths=None):
    if remaining_lengths is None:
        remaining_lengths = FLEET[:]  # uproszczenie

    heat = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    hits = set(map(tuple, np.argwhere(obs == 2)))

    for L in remaining_lengths:
        # poziomo
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE - L + 1):
                cells = [(r, c + i) for i in range(L)]
                if not _fits(cells, obs):
                    continue

                # jeśli są hity, preferuj ustawienia które je pokrywają
                cells_set = set(cells)
                if hits and len(hits & cells_set) == 0:
                    continue

                for rr, cc in cells:
                    if obs[rr, cc] == 0:
                        heat[rr, cc] += 1.0

        # pionowo
        for r in range(BOARD_SIZE - L + 1):
            for c in range(BOARD_SIZE):
                cells = [(r + i, c) for i in range(L)]
                if not _fits(cells, obs):
                    continue

                cells_set = set(cells)
                if hits and len(hits & cells_set) == 0:
                    continue

                for rr, cc in cells:
                    if obs[rr, cc] == 0:
                        heat[rr, cc] += 1.0

    # blokada skosów od hitów
    for hr, hc in hits:
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            rr, cc = hr + dr, hc + dc
            if 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and obs[rr, cc] == 0:
                heat[rr, cc] = 0.0

    return heat


def hybrid_pick_action(q_values, obs, legal_mask, alpha=0.55, beta=0.35):
    legal = np.where(legal_mask == 1)[0]
    if len(legal) == 0:
        return 0

    heat = build_probability_heatmap(obs)
    h = heat.reshape(-1)

    # target bonus N/S/E/W obok hitów
    target_bonus = np.zeros_like(h, dtype=np.float32)
    hits = set(map(tuple, np.argwhere(obs == 2)))
    for hr, hc in hits:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr, cc = hr + dr, hc + dc
            if 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and obs[rr, cc] == 0:
                target_bonus[rr * BOARD_SIZE + cc] += 1.0

    # norm
    q = q_values.astype(np.float32)
    qn = (q - q.min()) / (q.max() - q.min() + 1e-8)
    hn = (h - h.min()) / (h.max() - h.min() + 1e-8)
    tn = (target_bonus - target_bonus.min()) / (target_bonus.max() - target_bonus.min() + 1e-8) if target_bonus.max() > 0 else target_bonus

    score = qn + alpha * hn + beta * tn

    # legal mask
    masked = np.full_like(score, -1e9, dtype=np.float32)
    masked[legal] = score[legal]
    return int(np.argmax(masked))