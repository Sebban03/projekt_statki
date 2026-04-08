import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_module(module_name: str):
    cmd = [sys.executable, "-m", module_name]
    print(f"\n>>> Uruchamiam: {' '.join(cmd)}\n")
    subprocess.run(cmd, cwd=ROOT, check=False)


def main():
    while True:
        print("\n=== STATKI RL - MENU ===")
        print("1) Trening self-play")
        print("2) Ewaluacja modeli")
        print("3) Demo GUI: Gracz vs Model")
        print("4) Wykres treningu")
        print("5) Wyjście")

        choice = input("Wybierz opcję [1-5]: ").strip()

        if choice == "1":
            run_module("scripts.train_self_play")
        elif choice == "2":
            run_module("scripts.evaluate_models")
        elif choice == "3":
            run_module("scripts.play_human_vs_model_gui")
        elif choice == "4":
            run_module("scripts.plot_training")
        elif choice == "5":
            print("Koniec.")
            break
        else:
            print("Niepoprawny wybór.")


if __name__ == "__main__":
    main()