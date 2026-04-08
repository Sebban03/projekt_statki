import pandas as pd
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("checkpoints/train_log.csv")
    plt.figure(figsize=(10, 5))
    plt.plot(df["episode"], df["reward"], alpha=0.3, label="reward/episode")
    plt.plot(df["episode"], df["avg100"], linewidth=2, label="avg100")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training progress")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("checkpoints/training_plot.png", dpi=140)
    plt.show()

if __name__ == "__main__":
    main()