import numpy as np
import gymnasium as gym
from gymnasium import spaces

BOARD_SIZE = 10
FLEET = [4, 3, 3, 2, 2, 2, 1, 1, 1, 1]


def can_place(board, cells):
    for r, c in cells:
        if board[r, c] > 0:
            return False
        for rr in range(max(0, r - 1), min(BOARD_SIZE, r + 2)):
            for cc in range(max(0, c - 1), min(BOARD_SIZE, c + 2)):
                if board[rr, cc] > 0:
                    return False
    return True


def random_place_fleet(rng):
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int16)
    ship_id = 1
    ship_cells = {}
    ship_hits = {}

    for L in FLEET:
        placed = False
        for _ in range(5000):
            horiz = bool(rng.integers(0, 2))
            if horiz:
                r = rng.integers(0, BOARD_SIZE)
                c = rng.integers(0, BOARD_SIZE - L + 1)
                cells = [(r, c + i) for i in range(L)]
            else:
                r = rng.integers(0, BOARD_SIZE - L + 1)
                c = rng.integers(0, BOARD_SIZE)
                cells = [(r + i, c) for i in range(L)]

            if can_place(board, cells):
                for rr, cc in cells:
                    board[rr, cc] = ship_id
                ship_cells[ship_id] = set(cells)
                ship_hits[ship_id] = 0
                ship_id += 1
                placed = True
                break

        if not placed:
            raise RuntimeError("Nie udało się rozstawić floty.")

    return board, ship_cells, ship_hits


class BattleshipSingleAgentEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, seed=None, max_steps=100):
        super().__init__()
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self.observation_space = spaces.Box(low=0, high=2, shape=(BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.action_space = spaces.Discrete(BOARD_SIZE * BOARD_SIZE)

        self.hidden = None
        self.ship_cells = None
        self.ship_hits = None

        self.obs = None
        self.steps = 0

        # trafione pola statków, które nie są jeszcze zatopione
        self.hit_cells_alive = set()

    def _get_action_mask(self):
        base = (self.obs.reshape(-1) == 0).astype(np.int8)

        if not self.hit_cells_alive:
            return base

        # target mode: tylko sąsiedzi ortogonalni (plus), bez skosów
        target = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        for (r, c) in self.hit_cells_alive:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and self.obs[rr, cc] == 0:
                    target[rr, cc] = 1

        target_flat = target.reshape(-1)
        if target_flat.sum() > 0:
            return target_flat
        return base

    def _is_ship_sunk(self, ship_id):
        return self.ship_hits[ship_id] >= len(self.ship_cells[ship_id])

    def _mark_sunk_border_as_miss(self, ship_id):
        border = set()
        for (r, c) in self.ship_cells[ship_id]:
            for rr in range(max(0, r - 1), min(BOARD_SIZE, r + 2)):
                for cc in range(max(0, c - 1), min(BOARD_SIZE, c + 2)):
                    border.add((rr, cc))

        border -= self.ship_cells[ship_id]

        for (rr, cc) in border:
            if self.obs[rr, cc] == 0:
                self.obs[rr, cc] = 1

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.hidden, self.ship_cells, self.ship_hits = random_place_fleet(self.rng)
        self.obs = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.steps = 0
        self.hit_cells_alive = set()

        info = {"action_mask": self._get_action_mask()}
        return self.obs.copy(), info

    def step(self, action):
        self.steps += 1

        r, c = divmod(int(action), BOARD_SIZE)

        reward = 0.0
        terminated = False
        truncated = False

        if self.obs[r, c] != 0:
            reward = -1.0
            info = {"action_mask": self._get_action_mask()}
            if self.steps >= self.max_steps:
                truncated = True
            return self.obs.copy(), reward, terminated, truncated, info

        if self.hidden[r, c] > 0:
            ship_id = int(self.hidden[r, c])
            self.obs[r, c] = 2
            self.ship_hits[ship_id] += 1
            self.hit_cells_alive.add((r, c))

            reward = 1.5

            # bonus za kontynuację celu ortogonalnie
            neigh_hit = False
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and self.obs[rr, cc] == 2:
                    neigh_hit = True
                    break
            if neigh_hit:
                reward += 0.5

            if self._is_ship_sunk(ship_id):
                reward += 3.0

                for cell in self.ship_cells[ship_id]:
                    self.hit_cells_alive.discard(cell)

                self._mark_sunk_border_as_miss(ship_id)

            all_sunk = True
            for sid in self.ship_cells.keys():
                if self.ship_hits[sid] < len(self.ship_cells[sid]):
                    all_sunk = False
                    break
            if all_sunk:
                reward += 10.0
                terminated = True

        else:
            self.obs[r, c] = 1
            reward = -0.20

            # mocniejsza kara za strzał poza targetem, gdy są aktywne trafienia
            if self.hit_cells_alive:
                is_target = False
                for (hr, hc) in self.hit_cells_alive:
                    if abs(hr - r) + abs(hc - c) == 1:
                        is_target = True
                        break
                if not is_target:
                    reward -= 0.50

        if self.steps >= self.max_steps and not terminated:
            truncated = True

        info = {"action_mask": self._get_action_mask()}
        return self.obs.copy(), reward, terminated, truncated, info

    def render(self, mode="ansi"):
        if mode == "ansi":
            chars = {0: ".", 1: "o", 2: "x"}
            lines = []
            header = "   " + " ".join(list("ABCDEFGHIJ"))
            lines.append(header)
            for r in range(BOARD_SIZE):
                row = " ".join(chars[int(v)] for v in self.obs[r])
                lines.append(f"{r+1:2d} {row}")
            return "\n".join(lines)

        if mode == "human":
            print(self.render(mode="ansi"))