import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from envs.battleship_rl_env import BattleshipSingleAgentEnv, BOARD_SIZE
from scripts.heatmap_policy import hybrid_pick_action

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNetCNN(nn.Module):
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
        return self.head(self.features(x))


class ReplayBuffer:
    def __init__(self, capacity=200_000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, ns, done, mask, nmask):
        self.buf.append((s, a, r, ns, done, mask, nmask))

    def sample(self, batch_size=128):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d, m, nm = zip(*batch)
        return (
            np.array(s, dtype=np.float32),     # (B,10,10)
            np.array(a, dtype=np.int64),       # (B,)
            np.array(r, dtype=np.float32),     # (B,)
            np.array(ns, dtype=np.float32),    # (B,10,10)
            np.array(d, dtype=np.float32),     # (B,)
            np.array(m, dtype=np.int8),        # (B,100)
            np.array(nm, dtype=np.int8),       # (B,100)
        )

    def __len__(self):
        return len(self.buf)


def select_action(model, obs, action_mask, eps, alpha=0.55, beta=0.35):
    legal = np.where(action_mask == 1)[0]
    if len(legal) == 0:
        return 0

    # eksploracja
    if np.random.rand() < eps:
        return int(np.random.choice(legal))

    # Q + heatmapa (hybryda)
    with torch.no_grad():
        s = torch.tensor(obs[None, None, :, :], dtype=torch.float32, device=DEVICE)  # (1,1,10,10)
        q = model(s).cpu().numpy()[0]  # (100,)

    a = hybrid_pick_action(q_values=q, obs=obs, legal_mask=action_mask, alpha=alpha, beta=beta)
    return int(a)


def train():
    episodes = 3000
    batch_size = 128
    gamma = 0.99
    lr = 1e-4

    warmup_steps = 5000
    target_update_every = 1000
    train_every = 4

    epsilon_start = 1.0
    epsilon_end = 0.03
    epsilon_decay_eps = 3500

    # Wpływ heatmapy podczas wyboru akcji (eksploatacja)
    alpha = 0.55
    beta = 0.35

    os.makedirs("checkpoints", exist_ok=True)

    env = BattleshipSingleAgentEnv(max_steps=100)
    q_net = QNetCNN().to(DEVICE)
    target_net = QNetCNN().to(DEVICE)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    replay = ReplayBuffer(capacity=200_000)

    best_avg100 = -1e9
    reward_hist = []
    global_step = 0

    for ep in range(1, episodes + 1):
        obs, info = env.reset(seed=ep)
        done = False
        ep_reward = 0.0

        eps = max(
            epsilon_end,
            epsilon_start - (epsilon_start - epsilon_end) * (ep / epsilon_decay_eps),
        )

        while not done:
            a = select_action(q_net, obs, info["action_mask"], eps, alpha=alpha, beta=beta)
            nobs, r, terminated, truncated, ninfo = env.step(a)
            done_flag = float(terminated or truncated)

            replay.push(obs, a, r, nobs, done_flag, info["action_mask"], ninfo["action_mask"])

            obs = nobs
            info = ninfo
            ep_reward += r
            done = bool(done_flag)
            global_step += 1

            if len(replay) >= warmup_steps and global_step % train_every == 0:
                s, a_b, r_b, ns, d_b, m_b, nm_b = replay.sample(batch_size)

                s_t = torch.tensor(s[:, None, :, :], dtype=torch.float32, device=DEVICE)    # (B,1,10,10)
                ns_t = torch.tensor(ns[:, None, :, :], dtype=torch.float32, device=DEVICE)  # (B,1,10,10)
                a_t = torch.tensor(a_b, dtype=torch.int64, device=DEVICE)
                r_t = torch.tensor(r_b, dtype=torch.float32, device=DEVICE)
                d_t = torch.tensor(d_b, dtype=torch.float32, device=DEVICE)
                nm_t = torch.tensor(nm_b, dtype=torch.bool, device=DEVICE)  # (B,100)

                q_vals = q_net(s_t)                                # (B,100)
                q_sa = q_vals.gather(1, a_t.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    # Double DQN (bez heatmapy w targetach, klasycznie i stabilnie)
                    q_next_online = q_net(ns_t)                    # (B,100)
                    q_next_online[~nm_t] = -1e9
                    next_actions = torch.argmax(q_next_online, dim=1, keepdim=True)

                    q_next_target = target_net(ns_t)
                    q_next = q_next_target.gather(1, next_actions).squeeze(1)

                    target = r_t + (1.0 - d_t) * gamma * q_next

                loss = nn.functional.smooth_l1_loss(q_sa, target)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
                optimizer.step()

                if global_step % target_update_every == 0:
                    target_net.load_state_dict(q_net.state_dict())

        reward_hist.append(ep_reward)
        avg100 = float(np.mean(reward_hist[-100:]))

        if ep % 50 == 0:
            print(
                f"[EP {ep:4d}] reward={ep_reward:7.2f} avg100={avg100:7.2f} "
                f"eps={eps:5.3f} buffer={len(replay)}"
            )

        torch.save(q_net.state_dict(), "checkpoints/selfplay_latest.pt")
        if ep >= 100 and avg100 > best_avg100:
            best_avg100 = avg100
            torch.save(q_net.state_dict(), "checkpoints/selfplay_best.pt")

    print("Koniec treningu.")
    print(f"Najlepsze avg100: {best_avg100:.3f}")


if __name__ == "__main__":
    train()