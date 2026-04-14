# Sprawdza, czy seedy treningu i testu się pokrywają

def build_train_seeds(episodes: int):
    # tak jak w train_self_play.py: reset(seed=ep)
    return set(range(1, episodes + 1))


def build_eval_seeds(episodes: int, seed_base: int):
    # tak jak w evaluate_models.py: reset(seed=seed_base + ep)
    return set(seed_base + i for i in range(episodes))


def main():
    # Dostosuj pod swój case:
    train_episodes = 3000
    eval_episodes = 1000
    eval_seed_base = 10000

    train_seeds = build_train_seeds(train_episodes)
    eval_seeds = build_eval_seeds(eval_episodes, eval_seed_base)

    overlap = train_seeds & eval_seeds

    print("=== Seed split checker ===")
    print(f"Train episodes: {train_episodes}")
    print(f"Eval episodes:  {eval_episodes}")
    print(f"Eval seed base: {eval_seed_base}")
    print(f"Train seed range: {min(train_seeds)}..{max(train_seeds)}")
    print(f"Eval seed range:  {min(eval_seeds)}..{max(eval_seeds)}")
    print(f"Overlap count: {len(overlap)}")

    if overlap:
        sample = sorted(list(overlap))[:20]
        print("UWAGA: seedy się pokrywają! Przykład:", sample)
    else:
        print("OK: brak pokrycia seedów (uczciwy split train/test).")


if __name__ == "__main__":
    main()