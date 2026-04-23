# Statki (Battleship) RL: Double DQN + Heatmap Heuristics

Projekt implementujący inteligentnego agenta do gry w Statki (Battleship), wykorzystującego uczenie ze wzmocnieniem (Reinforcement Learning) połączone z probabilistyczną heurystyką (Heatmap). Agent został wytrenowany przy użyciu algorytmu **Double Deep Q-Network (DDQN)** z konwolucyjną siecią neuronową (CNN).

Dodatkowo projekt zawiera w pełni funkcjonalny, interaktywny interfejs graficzny (Pygame) w pirackim klimacie, pozwalający na grę człowiek vs model AI.

---

## Struktura i kluczowe funkcjonalności (Pod Zaliczenie)

Poniżej znajduje się zestawienie najważniejszych mechanizmów wymaganych do zaliczenia projektu ze wskazaniem, w których plikach się znajdują:

### 1. Autorskie Środowisko Gym (Zasady Gry i Maskowanie Akcji)
**Plik:** `envs/battleship_rl_env.py`
- Zaimplementowano środowisko oparte na standardzie `gymnasium.Env`.
- **Action Masking:** Funkcja `_get_action_mask()` gwarantuje, że agent wybiera tylko legalne ruchy (nie strzela dwa razy w to samo miejsce).
- **Target Mode:** Maskowanie wymusza strzelanie w "plusie" (góra/dół/lewo/prawo) po trafieniu, blokując skosy.
- **Smart Border:** Funkcja `_mark_sunk_border_as_miss()` automatycznie oznacza obwódkę zatopionego statku jako pudła, eliminując bezsensowne strzały agenta.

### 2. Reward Shaping (Kształtowanie Nagród)
**Plik:** `envs/battleship_rl_env.py` (funkcja `step`)
Agent uczy się na bazie starannie dobranej funkcji nagrody:
- **Pudło:** `-0.20` (kara wymuszająca minimalizację ruchów).
- **Pudło poza celem (gdy statek jest trafiony, ale agent strzeli gdzie indziej):** `-0.50`.
- **Trafienie:** `+1.50`.
- **Ortogonalne kontynuowanie trafienia:** `+0.50` (nagroda za logiczne myślenie).
- **Zatopienie statku:** `+3.00`.
- **Wygrana (zatopienie całej floty):** `+10.00`.

### 3. Architektura Sieci (CNN) i Algorytm RL (Double DQN)
**Plik:** `scripts/train_self_play.py`
- Zastosowano **Konwolucyjną Sieć Neuronową (`QNetCNN`)** z trzema warstwami `Conv2d`, idealną do analizy siatki przestrzennej 10x10.
- Agent trenuje się wykorzystując **Replay Buffer** oraz algorytm **Double DQN** (osobna sieć online i target network, stabilizujące proces uczenia, równanie Bellmana).

### 4. Polityka Hybrydowa (RL + Probabilistyka)
**Plik:** `scripts/heatmap_policy.py`
- Zaimplementowano zaawansowaną heurystykę `build_probability_heatmap()`, która w czasie rzeczywistym oblicza gęstość prawdopodobieństwa wystąpienia statków na nieodkrytych polach (Baysian-like approach).
- Funkcja `hybrid_pick_action()` łączy wyjście z sieci neuronowej (Q-values) z matematyczną heatmapą. Model nie musi uczyć się rozkładu na pamięć, lecz "wspomaga" się statystyką do podejmowania perfekcyjnych decyzji (w ewaluacji agent kończy grę średnio w ~58 ruchach).

### 5. Walidacja, Testowanie i Seed Splitting
**Pliki:** `scripts/evaluate_models.py`, `scripts/check_seed_split.py`
- Rygorystyczny podział na zbiór treningowy i testowy zapobiega zjawisku *overfittingu*. `check_seed_split.py` weryfikuje brak nakładania się ziarna losowości (seedów) między mapami uczącymi a testowymi.
- `evaluate_models.py` pozwala na przeprowadzenie masowych testów (np. 1000 gier) z generowaniem precyzyjnych raportów statystycznych (Win-rate, Avg reward, Avg episode length).

### 6. Wizualizacja i Analiza Treningu
**Plik:** `scripts/plot_training.py`
- Generowanie wykresów z przebiegu uczenia (`reward` i `avg100` per epizod) za pomocą biblioteki *matplotlib* oraz *pandas*, eksportowanych do pliku `training_plot.png`.

### 7. Interfejs Graficzny GUI (Człowiek vs AI)
**Plik:** `scripts/play_human_vs_model_gui.py`
- W pełni grywalna aplikacja napisana w `Pygame`.
- Moduł `AudioSystem` obsługuje muzykę tła, playlisty bitewne oraz efekty dźwiękowe (SFX) strzałów, pudeł i zatopień.
- Animacje cząsteczkowe na planszy, liczniki ruchów i tur, dynamiczne zmiany faz gry i graficzne komunikaty (Overlay) o zwycięstwie/porażce.

---

## Jak uruchomić projekt?

**WAŻNE:** Wagi wytrenowanego modelu nie znajdują się bezpośrednio w repozytorium ze względu na swój rozmiar. 
1. Pobierz plik `checkpoints.zip` (z zakładki Releases / z załącznika).
2. Wypakuj go tak, aby w głównym katalogu projektu powstał folder `checkpoints/` zawierający pliki `.pt` oraz `.json`.

Uruchom główny plik sterujący:
```bash
python run_app.py