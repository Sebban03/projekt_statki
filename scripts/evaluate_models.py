import os
import json
import numpy as np
import torch
import torch.nn as nn

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


def pick_action_eval(model, obs, mask, alpha=0.55, beta=0.35):
    legal = np.where(mask == 1)[0]
    if len(legal) == 0:
        return 0

    with torch.no_grad():
        s = torch.tensor(obs[None, None, :, :], dtype=torch.float32, device=DEVICE)
        q = model(s).cpu().numpy()[0]

    return int(hybrid_pick_action(q_values=q, obs=obs, legal_mask=mask, alpha=alpha, beta=beta))


def run_eval(model_path, episodes=1000, seed_base=10000, max_steps=100, alpha=0.55, beta=0.35):
    if not os.path.exists(model_path):
        return {"model_path": model_path, "error": "missing_file"}

    model = QNetCNN().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    env = BattleshipSingleAgentEnv(max_steps=max_steps)

    rewards = []
    wins = 0
    lengths = []

    for ep in range(episodes):
        obs, info = env.reset(seed=seed_base + ep)
        done = False
        ep_reward = 0.0
        ep_len = 0
        terminated_flag = False

        while not done:
            a = pick_action_eval(model, obs, info["action_mask"], alpha=alpha, beta=beta)
            obs, r, terminated, truncated, info = env.step(a)
            ep_reward += r
            ep_len += 1
            done = terminated or truncated
            terminated_flag = terminated

        rewards.append(ep_reward)
        lengths.append(ep_len)
        if terminated_flag:
            wins += 1

    rewards = np.array(rewards, dtype=np.float32)
    lengths = np.array(lengths, dtype=np.float32)

    return {
        "model_path": model_path,
        "episodes": episodes,
        "seed_base": seed_base,
        "win_rate": float(wins / episodes),
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "avg_episode_len": float(np.mean(lengths)),
        "std_episode_len": float(np.std(lengths)),
        "alpha": alpha,
        "beta": beta,
    }


def eval_multi_seed(model_path, episodes_per_seed=200, seed_bases=(10000, 20000, 30000, 40000, 50000), max_steps=100, alpha=0.55, beta=0.35):
    runs = []
    for sb in seed_bases:
        runs.append(run_eval(model_path, episodes=episodes_per_seed, seed_base=sb, max_steps=max_steps, alpha=alpha, beta=beta))

    valid_runs = [r for r in runs if "error" not in r]
    if not valid_runs:
        return {"model_path": model_path, "error": "all_runs_failed", "runs": runs}

    win_rates = np.array([r["win_rate"] for r in valid_runs], dtype=np.float32)
    avg_rewards = np.array([r["avg_reward"] for r in valid_runs], dtype=np.float32)
    avg_lens = np.array([r["avg_episode_len"] for r in valid_runs], dtype=np.float32)

    return {
        "model_path": model_path,
        "episodes_total": int(episodes_per_seed * len(valid_runs)),
        "seeds_tested": [int(r["seed_base"]) for r in valid_runs],
        "win_rate_mean": float(np.mean(win_rates)),
        "win_rate_std": float(np.std(win_rates)),
        "avg_reward_mean": float(np.mean(avg_rewards)),
        "avg_reward_std": float(np.std(avg_rewards)),
        "avg_len_mean": float(np.mean(avg_lens)),
        "avg_len_std": float(np.std(avg_lens)),
        "alpha": alpha,
        "beta": beta,
        "runs": runs,
    }


def print_single(result):
    if "error" in result:
        print(f"\nModel: {result['model_path']}\nERROR: {result['error']}")
        return
    print(f"\nModel: {result['model_path']}")
    print(f"Episodes: {result['episodes']}")
    print(f"Seed base: {result['seed_base']}")
    print(f"Win-rate: {result['win_rate']:.3f}")
    print(f"Avg reward: {result['avg_reward']:.2f}")
    print(f"Std reward: {result['std_reward']:.2f}")
    print(f"Min/Max reward: {result['min_reward']:.2f} / {result['max_reward']:.2f}")
    print(f"Avg episode len: {result['avg_episode_len']:.2f} ± {result['std_episode_len']:.2f}")
    print(f"Policy mix: alpha={result['alpha']}, beta={result['beta']}")


def print_multi(summary):
    if "error" in summary:
        print(f"\nModel: {summary['model_path']}\nERROR: {summary['error']}")
        return

    print(f"\n=== MULTI-SEED SUMMARY ===")
    print(f"Model: {summary['model_path']}")
    print(f"Episodes total: {summary['episodes_total']}")
    print(f"Seeds: {summary['seeds_tested']}")
    print(f"Win-rate mean ± std: {summary['win_rate_mean']:.3f} ± {summary['win_rate_std']:.3f}")
    print(f"Avg reward mean ± std: {summary['avg_reward_mean']:.2f} ± {summary['avg_reward_std']:.2f}")
    print(f"Avg episode len mean ± std: {summary['avg_len_mean']:.2f} ± {summary['avg_len_std']:.2f}")
    print(f"Policy mix: alpha={summary['alpha']}, beta={summary['beta']}")

    print("\nPer-seed:")
    for r in summary["runs"]:
        if "error" in r:
            print(f"  seed=? -> ERROR: {r['error']}")
        else:
            print(f"  seed={r['seed_base']}: win={r['win_rate']:.3f}, reward={r['avg_reward']:.2f}, len={r['avg_episode_len']:.2f}")


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    alpha = 0.55
    beta = 0.35

    single_best = run_eval("checkpoints/selfplay_best.pt", episodes=1000, seed_base=10000, max_steps=100, alpha=alpha, beta=beta)
    single_latest = run_eval("checkpoints/selfplay_latest.pt", episodes=1000, seed_base=10000, max_steps=100, alpha=alpha, beta=beta)

    print_single(single_best)
    print_single(single_latest)

    multi_best = eval_multi_seed("checkpoints/selfplay_best.pt", episodes_per_seed=200, alpha=alpha, beta=beta)
    multi_latest = eval_multi_seed("checkpoints/selfplay_latest.pt", episodes_per_seed=200, alpha=alpha, beta=beta)

    print_multi(multi_best)
    print_multi(multi_latest)

    save_json("checkpoints/eval_single_best.json", single_best)
    save_json("checkpoints/eval_single_latest.json", single_latest)
    save_json("checkpoints/eval_multi_best.json", multi_best)
    save_json("checkpoints/eval_multi_latest.json", multi_latest)
    print("\nZapisano raporty JSON w folderze checkpoints/.")