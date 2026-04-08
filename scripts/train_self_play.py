import os
import csv
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from envs.battleship_rl_env import BattleshipSingleAgentEnv, BOARD_SIZE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetCNN(nn.Module):
    """
    Wejście: (B, 1, 10, 10)
    Kanał zawiera:
      0 = nieznane
      1 = pudło
      2 = trafienie
    Wyjście: Q-values dla 100 akcji.
    """
    def __init__(self, out_dim=BOARD_SIZE * BOARD_SIZE):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * BOARD_SIZE * BOARD_SIZE, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim),
        )

    def forward(self, x):
        x = self.features(x)
        return self.head(x)


class ReplayBuffer:
    def __init__(self, capacity=150_000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, d, m2):
        self.buf.append((s, a, r, s2, d, m2))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d, m2 = zip(*batch)
        return (
            np.array(s, dtype=np.float32),    # (B, 10, 10)
            np.array(a, dtype=np.int64),      # (B,)
            np.array(r, dtype=np.float32),    # (B,)
            np.array(s2, dtype=np.float32),   # (B, 10, 10)
            np.array(d, dtype=np.float32),    # (B,)
            np.array(m2, dtype=np.int8),      # (B, 100)
        )

    def __len__(self):
        return len(self.buf)


def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)


def soft_update(target, source, tau=0.01):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tp.data * (1.0 - tau) + sp.data * tau)


def select_action(model, state, mask, epsilon):
    legal = np.where(mask == 1)[0]
    if len(legal) == 0:
        return 0

    if random.random() < epsilon:
        return int(np.random.choice(legal))

    with torch.no_grad():
        s = torch.tensor(state[None, None, :, :], dtype=torch.float32, device=DEVICE)  # (1,1,10,10)
        q = model(s).cpu().numpy()[0]  # (100,)
        q_masked = np.full_like(q, -1e9, dtype=np.float32)
        q_masked[legal] = q[legal]
        return int(np.argmax(q_masked))


def train():
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("checkpoints/opponents", exist_ok=True)

    log_path = "checkpoints/train_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode", "reward", "avg100", "epsilon", "buffer_size"])

    env = BattleshipSingleAgentEnv(seed=42, max_steps=100)

    qnet = QNetCNN().to(DEVICE)
    target = QNetCNN().to(DEVICE)
    target.load_state_dict(qnet.state_dict())

    optimizer = optim.Adam(qnet.parameters(), lr=5e-4)
    replay = ReplayBuffer(capacity=150_000)

    gamma = 0.99
    batch_size = 128
    warmup = 6000
    total_episodes = 5000
    train_every = 1
    tau = 0.01

    epsilon_start, epsilon_end = 1.0, 0.03
    epsilon_decay_eps = 3500

    step_count = 0
    best_avg = -1e9
    rewards_window = deque(maxlen=100)
    opponent_paths = []

    huber = nn.SmoothL1Loss()

    for ep in range(1, total_episodes + 1):
        state, info = env.reset(seed=1000 + ep * 11)
        done = False
        ep_reward = 0.0

        epsilon = max(
            epsilon_end,
            epsilon_start - (epsilon_start - epsilon_end) * (ep / epsilon_decay_eps)
        )

        while not done:
            action = select_action(qnet, state, info["action_mask"], epsilon)
            next_state, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated

            replay.push(state, action, reward, next_state, done, next_info["action_mask"])

            state = next_state
            info = next_info
            ep_reward += reward
            step_count += 1

            if len(replay) >= warmup and step_count % train_every == 0:
                s, a, r, s2, d, m2 = replay.sample(batch_size)

                s_t = torch.tensor(s[:, None, :, :], dtype=torch.float32, device=DEVICE)   # (B,1,10,10)
                s2_t = torch.tensor(s2[:, None, :, :], dtype=torch.float32, device=DEVICE) # (B,1,10,10)
                a_t = torch.tensor(a, dtype=torch.long, device=DEVICE).unsqueeze(1)        # (B,1)
                r_t = torch.tensor(r, dtype=torch.float32, device=DEVICE).unsqueeze(1)      # (B,1)
                d_t = torch.tensor(d, dtype=torch.float32, device=DEVICE).unsqueeze(1)      # (B,1)
                m2_t = torch.tensor(m2, dtype=torch.bool, device=DEVICE)                    # (B,100)

                # Q(s,a)
                q_pred = qnet(s_t).gather(1, a_t)

                # Double DQN + legal mask na next actions
                with torch.no_grad():
                    q_next_online = qnet(s2_t)  # (B,100)
                    q_next_online_masked = q_next_online.masked_fill(~m2_t, -1e9)
                    next_actions = q_next_online_masked.argmax(dim=1, keepdim=True)  # (B,1)

                    q_next_target = target(s2_t).gather(1, next_actions)  # (B,1)
                    y = r_t + gamma * (1.0 - d_t) * q_next_target

                loss = huber(q_pred, y)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(qnet.parameters(), max_norm=10.0)
                optimizer.step()

                soft_update(target, qnet, tau=tau)

        rewards_window.append(ep_reward)
        avg100 = float(np.mean(rewards_window))

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([ep, ep_reward, avg100, epsilon, len(replay)])

        if ep % 50 == 0:
            print(
                f"[EP {ep:4d}] reward={ep_reward:7.2f} avg100={avg100:7.2f} "
                f"eps={epsilon:.3f} buffer={len(replay)}"
            )

        if ep % 25 == 0:
            save_checkpoint(qnet, "checkpoints/selfplay_latest.pt")

        if ep >= 300 and avg100 > best_avg:
            best_avg = avg100
            save_checkpoint(qnet, "checkpoints/selfplay_best.pt")
            opp_path = f"checkpoints/opponents/opp_ep{ep}_avg{avg100:.2f}.pt"
            save_checkpoint(qnet, opp_path)
            opponent_paths.append(opp_path)

    save_checkpoint(qnet, "checkpoints/selfplay_final.pt")
    print("Trening zakończony.")
    print(f"Best avg100: {best_avg:.2f}")
    print(f"Opponents saved: {len(opponent_paths)}")
    print(f"Log CSV: {log_path}")


if __name__ == "__main__":
    train()