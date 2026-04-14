import os
import random
import pygame
import numpy as np
import torch
import torch.nn as nn

from envs.battleship_rl_env import BOARD_SIZE, FLEET
from scripts.heatmap_policy import hybrid_pick_action

# =========================
# USTAWIENIA / KONFIG
# =========================
CELL = 36
MARGIN = 30
BOARD_PIX = BOARD_SIZE * CELL
STATUS_H = 160
WIDTH = 2 * BOARD_PIX + 3 * MARGIN
HEIGHT = 740  # Sztywna wysokość okna, żeby wszystko się swobodnie zmieściło

AI_COOLDOWN_MS = 1000  # 1s przerwy między ruchami AI

ICON_PATH = "assets/ui/pirate_icon.png"

# =========================
# MODEL
# =========================
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
        x = self.features(x)
        return self.head(x)


# =========================
# LOGIKA PLANSZY
# =========================
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
            h = bool(rng.integers(0, 2))
            if h:
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


def board_signature(board):
    return (board > 0).astype(np.uint8).tobytes()


def generate_unique_player_board(rng, seen_signatures, max_tries=200):
    for _ in range(max_tries):
        b, cells, hits = random_place_fleet(rng)
        sig = board_signature(b)
        if sig not in seen_signatures:
            seen_signatures.add(sig)
            return b, cells, hits, False
    b, cells, hits = random_place_fleet(rng)
    return b, cells, hits, True


def all_sunk(ship_cells, ship_hits):
    for sid, cells in ship_cells.items():
        if ship_hits[sid] < len(cells):
            return False
    return True


def mark_sunk_border_as_miss(board_view, ship_cells_set):
    border = set()
    for (r, c) in ship_cells_set:
        for rr in range(max(0, r - 1), min(BOARD_SIZE, r + 2)):
            for cc in range(max(0, c - 1), min(BOARD_SIZE, c + 2)):
                border.add((rr, cc))
    border -= ship_cells_set

    for (rr, cc) in border:
        if board_view[rr, cc] == 0:
            board_view[rr, cc] = 1


def model_action(model, obs, legal_mask, alpha=0.55, beta=0.35):
    legal = np.where(legal_mask == 1)[0]
    if len(legal) == 0:
        return 0

    with torch.no_grad():
        s = torch.tensor(obs[None, None, :, :], dtype=torch.float32)
        q = model(s).numpy()[0]

    return int(hybrid_pick_action(q_values=q, obs=obs, legal_mask=legal_mask, alpha=alpha, beta=beta))


# =========================
# UI HELPERY I RYSOWANIE
# =========================
def draw_x(screen, rect, color=(190, 20, 20), thickness=3):
    pygame.draw.line(screen, color, (rect.left + 6, rect.top + 6), (rect.right - 6, rect.bottom - 6), thickness)
    pygame.draw.line(screen, color, (rect.left + 6, rect.bottom - 6), (rect.right - 6, rect.top + 6), thickness)


def draw_grid_labels(screen, x0, y0, font, color=(25, 25, 25)):
    cols = "ABCDEFGHIJ"
    for c in range(BOARD_SIZE):
        txt = font.render(cols[c], True, color)
        tx = x0 + c * CELL + CELL // 2 - txt.get_width() // 2
        ty = y0 - 25
        screen.blit(txt, (tx, ty))
    for r in range(BOARD_SIZE):
        txt = font.render(str(r + 1), True, color)
        tx = x0 - 25
        ty = y0 + r * CELL + CELL // 2 - txt.get_height() // 2
        screen.blit(txt, (tx, ty))


def draw_board(screen, ships_board, shots_taken, ship_cells, ship_hits, x0, y0, show_ships=False):
    WATER = (187, 214, 236)
    WATER_ALT = (177, 206, 229)
    MISS = (132, 132, 132)
    SHIP = (64, 110, 170)
    SUNK = (176, 38, 38)
    HIT_BG = (236, 236, 236)
    GRID = (55, 55, 55)

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            rect = pygame.Rect(x0 + c * CELL, y0 + r * CELL, CELL, CELL)
            ship_id = int(ships_board[r, c])
            shot = int(shots_taken[r, c])

            base = WATER if (r + c) % 2 == 0 else WATER_ALT
            color = base
            draw_hit_x = False

            if shot == 1:
                color = MISS
            elif shot == 2:
                if ship_id > 0:
                    length = len(ship_cells[ship_id])
                    hits = ship_hits[ship_id]
                    if hits >= length:
                        color = SUNK
                    else:
                        color = HIT_BG
                        draw_hit_x = True
                else:
                    color = MISS
            else:
                if ship_id > 0 and show_ships:
                    color = SHIP

            pygame.draw.rect(screen, color, rect, border_radius=2)
            pygame.draw.rect(screen, GRID, rect, 1, border_radius=2)

            if draw_hit_x:
                draw_x(screen, rect)


def wrap_text(text, max_chars=95):
    words = text.split()
    lines, cur = [], []
    cur_len = 0
    for w in words:
        add = len(w) + (1 if cur else 0)
        if cur_len + add <= max_chars:
            cur.append(w)
            cur_len += add
        else:
            lines.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
    if cur:
        lines.append(" ".join(cur))
    return lines


def draw_button(screen, rect, text, font, bg=(70, 130, 180), fg=(255, 255, 255), border=(28, 28, 28)):
    pygame.draw.rect(screen, bg, rect, border_radius=10)
    pygame.draw.rect(screen, border, rect, 2, border_radius=10)
    t = font.render(text, True, fg)
    screen.blit(t, (rect.centerx - t.get_width() // 2, rect.centery - t.get_height() // 2))


def draw_wave_bg(screen):
    top = (214, 229, 242)
    bot = (198, 220, 238)
    w, h = screen.get_size()
    for y in range(h):
        t = y / max(1, h - 1)
        r = int(top[0] * (1 - t) + bot[0] * t)
        g = int(top[1] * (1 - t) + bot[1] * t)
        b = int(top[2] * (1 - t) + bot[2] * t)
        pygame.draw.line(screen, (r, g, b), (0, y), (w, y))


# =========================
# SYSTEM AUDIO (MUZYKA I SFX)
# =========================
class AudioSystem:
    def __init__(self):
        self.enabled = True
        self.sfx = {}
        self.menu_music = "assets/music/menu.mp3"
        self.game_playlist = []
        self.current_track_idx = 0
        self.state = "stopped"  # "menu", "game", "stopped"

        try:
            pygame.mixer.init()

            sfx_files = {
                "hit": "assets/sfx/hit.wav",
                "miss": "assets/sfx/miss.wav",
                "sunk": "assets/sfx/sunk.wav",
                "win": "assets/sfx/win.wav",
                "lose": "assets/sfx/lose.wav"
            }
            for name, path in sfx_files.items():
                if os.path.exists(path):
                    self.sfx[name] = pygame.mixer.Sound(path)
                    self.sfx[name].set_volume(0.6)

            if os.path.exists("assets/music"):
                for f in os.listdir("assets/music"):
                    if f.startswith("game") and (f.endswith(".mp3") or f.endswith(".ogg")):
                        self.game_playlist.append(os.path.join("assets/music", f))
                random.shuffle(self.game_playlist)

            self.MUSIC_END_EVENT = pygame.USEREVENT + 1
            pygame.mixer.music.set_endevent(self.MUSIC_END_EVENT)

        except Exception as e:
            print("Błąd inicjalizacji audio:", e)
            self.enabled = False

    def toggle(self):
        self.enabled = not self.enabled
        if not pygame.mixer.get_init():
            return self.enabled

        if self.enabled:
            pygame.mixer.music.unpause()
            if not pygame.mixer.music.get_busy() and self.state != "stopped":
                if self.state == "menu":
                    self.play_menu()
                elif self.state == "game":
                    self.play_next_game_track()
        else:
            pygame.mixer.music.pause()
        return self.enabled

    def play_sfx(self, name):
        if self.enabled and name in self.sfx:
            self.sfx[name].play()

    def play_menu(self):
        self.state = "menu"
        if not self.enabled or not pygame.mixer.get_init():
            return
        if os.path.exists(self.menu_music):
            pygame.mixer.music.load(self.menu_music)
            pygame.mixer.music.set_volume(0.3)
            pygame.mixer.music.play(-1)

    def play_game(self):
        self.state = "game"
        self.current_track_idx = 0
        if self.game_playlist:
            random.shuffle(self.game_playlist)
            self.play_next_game_track()

    def play_next_game_track(self):
        if not self.enabled or not pygame.mixer.get_init() or not self.game_playlist:
            return
        track = self.game_playlist[self.current_track_idx]
        pygame.mixer.music.load(track)
        pygame.mixer.music.set_volume(0.2)
        pygame.mixer.music.play()
        self.current_track_idx = (self.current_track_idx + 1) % len(self.game_playlist)

    def stop_music(self):
        self.state = "stopped"
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()

    def handle_event(self, event):
        if event.type == self.MUSIC_END_EVENT and self.state == "game" and self.enabled:
            self.play_next_game_track()


# =========================
# SYSTEM ANIMACJI
# =========================
animations = []


def add_anim(r, c, offset_x, offset_y, anim_type):
    cx = offset_x + c * CELL + CELL // 2
    cy = offset_y + r * CELL + CELL // 2
    animations.append({"x": cx, "y": cy, "type": anim_type, "life": 20, "max_life": 20})


def update_and_draw_anims(screen):
    for anim in animations:
        prog = 1.0 - (anim["life"] / anim["max_life"])
        surf = pygame.Surface((CELL * 3, CELL * 3), pygame.SRCALPHA)
        if anim["type"] == "miss":
            pygame.draw.circle(surf, (200, 200, 200, int(255 * (1 - prog))), (CELL * 1.5, CELL * 1.5),
                               int(CELL / 4 + prog * CELL))
        elif anim["type"] == "hit":
            pygame.draw.circle(surf, (255, 120, 0, int(255 * (1 - prog))), (CELL * 1.5, CELL * 1.5),
                               int(CELL / 4 + prog * CELL))
        elif anim["type"] == "sunk":
            pygame.draw.circle(surf, (220, 20, 20, int(255 * (1 - prog))), (CELL * 1.5, CELL * 1.5),
                               int(CELL / 2 + prog * CELL * 1.5))
        screen.blit(surf, (anim["x"] - CELL * 1.5, anim["y"] - CELL * 1.5))
        anim["life"] -= 1
    animations[:] = [a for a in animations if a["life"] > 0]


# =========================
# MAIN
# =========================
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Statki: Piraci z Karaibów")

    if os.path.exists(ICON_PATH):
        try:
            icon = pygame.image.load(ICON_PATH)
            pygame.display.set_icon(icon)
        except Exception:
            pass

    font = pygame.font.SysFont("georgia", 20)
    small = pygame.font.SysFont("georgia", 16)
    big = pygame.font.SysFont("georgia", 30, bold=True)
    huge = pygame.font.SysFont("georgia", 50, bold=True)
    clock = pygame.time.Clock()

    model = QNetCNN()
    model.load_state_dict(torch.load("checkpoints/selfplay_best.pt", map_location="cpu"))
    model.eval()

    alpha = 0.55
    beta = 0.35

    # System Audio
    audio = AudioSystem()
    audio.play_menu()

    rng = np.random.default_rng()
    seen_layouts = set()

    p_board = p_cells = p_hits = None
    ai_board = ai_cells = ai_hits = None
    p_shots_taken = ai_shots_taken = None
    p_view_on_ai = ai_view_on_p = None

    turn = "human"
    game_over = False
    msg = "Przygotuj flotę, Kapitanie! Wylosuj ustawienie okrętów."
    phase = "setup"
    ai_ready_time = 0

    # Liczniki
    human_moves = 0
    ai_moves = 0
    total_turns = 0

    # Współrzędne i rozstawienie (poprawione pod kątem skalowania)
    left_x = MARGIN
    right_x = 2 * MARGIN + BOARD_PIX

    # Przesuwamy planszę w dół, żeby opisy A-J i teksty miały wolne miejsce
    y0 = 135

    # Przyciski w setup (równo rozłożone poniżej planszy)
    btn_y = y0 + BOARD_PIX + 45
    btn_random = pygame.Rect(MARGIN, btn_y, 210, 46)
    btn_start = pygame.Rect(MARGIN + 230, btn_y, 160, 46)
    btn_new_pool = pygame.Rect(MARGIN + 410, btn_y, 160, 46)
    btn_music_setup = pygame.Rect(MARGIN + 590, btn_y, 160, 46)

    # Przycisk restart po końcu gry
    btn_restart = pygame.Rect(WIDTH // 2 - 125, HEIGHT // 2 + 60, 250, 60)

    def start_match():
        nonlocal ai_board, ai_cells, ai_hits
        nonlocal p_shots_taken, ai_shots_taken, p_view_on_ai, ai_view_on_p
        nonlocal turn, game_over, ai_ready_time, phase
        nonlocal human_moves, ai_moves, total_turns, msg

        ai_board, ai_cells, ai_hits = random_place_fleet(rng)

        p_shots_taken = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        ai_shots_taken = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        p_view_on_ai = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        ai_view_on_p = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)

        turn = "human"
        game_over = False
        phase = "game"
        ai_ready_time = 0

        human_moves = 0
        ai_moves = 0
        total_turns = 0
        animations.clear()
        msg = "Bitwa rozpoczęta! Twój ruch, Kapitanie."

        audio.play_game()

    while True:
        now_ms = pygame.time.get_ticks()

        # Dynamiczny Przycisk muzyki w grze
        status_y = y0 + BOARD_PIX + 30
        btn_music_game = pygame.Rect(WIDTH - MARGIN - 165, status_y + 15, 165, 46)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            audio.handle_event(event)

            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos

                if phase == "setup":
                    if btn_random.collidepoint(event.pos):
                        p_board, p_cells, p_hits, repeated = generate_unique_player_board(rng, seen_layouts)
                        msg = "Nowe ustawienie floty gotowe." if not repeated else "Ustawienie wylosowane (możliwa powtórka)."

                    elif btn_start.collidepoint(event.pos):
                        if p_board is None:
                            msg = "Najpierw przygotuj flotę (Wylosuj)!"
                        else:
                            start_match()

                    elif btn_new_pool.collidepoint(event.pos):
                        seen_layouts.clear()
                        p_board = p_cells = p_hits = None
                        msg = "Wyczyszczono pulę. Wylosuj nowe ustawienie."

                    elif btn_music_setup.collidepoint(event.pos):
                        audio.toggle()

                elif phase == "game" and not game_over:
                    if btn_music_game.collidepoint(event.pos):
                        audio.toggle()
                        continue

                    if turn == "human" and right_x <= mx < right_x + BOARD_PIX and y0 <= my < y0 + BOARD_PIX:
                        c = (mx - right_x) // CELL
                        r = (my - y0) // CELL

                        if ai_shots_taken[r, c] != 0:
                            msg = "Kule tam już leciały! Wybierz inne pole."
                        else:
                            human_moves += 1
                            total_turns += 1

                            if ai_board[r, c] > 0:
                                sid = int(ai_board[r, c])
                                ai_shots_taken[r, c] = 2
                                p_view_on_ai[r, c] = 2
                                ai_hits[sid] += 1

                                if ai_hits[sid] == len(ai_cells[sid]):
                                    msg = f"Zatopiliśmy wrogi okręt ({len(ai_cells[sid])}-maszt)! Ognia znowu!"
                                    mark_sunk_border_as_miss(p_view_on_ai, ai_cells[sid])
                                    audio.play_sfx("sunk")
                                    add_anim(r, c, right_x, y0, "sunk")
                                else:
                                    msg = "Trafiony! Okręt płonie!"
                                    audio.play_sfx("hit")
                                    add_anim(r, c, right_x, y0, "hit")
                            else:
                                ai_shots_taken[r, c] = 1
                                p_view_on_ai[r, c] = 1
                                msg = "Pudło. Kule wpadły do wody."
                                audio.play_sfx("miss")
                                add_anim(r, c, right_x, y0, "miss")
                                turn = "ai"
                                ai_ready_time = now_ms + AI_COOLDOWN_MS

                            if all_sunk(ai_cells, ai_hits):
                                game_over = True
                                msg = "ZWYCIĘSTWO! Morska dominacja należy do nas."
                                audio.stop_music()
                                audio.play_sfx("win")

                elif game_over:
                    if btn_restart.collidepoint(event.pos):
                        phase = "setup"
                        game_over = False
                        p_board = p_cells = p_hits = None
                        seen_layouts.clear()
                        msg = "Przygotuj flotę do nowej bitwy!"
                        animations.clear()
                        audio.play_menu()

        # RUCH AI
        if phase == "game" and not game_over and turn == "ai" and now_ms >= ai_ready_time:
            obs = ai_view_on_p.astype(np.float32)
            legal_mask = (ai_view_on_p.reshape(-1) == 0).astype(np.int8)

            action = model_action(model, obs, legal_mask, alpha=alpha, beta=beta)
            r, c = divmod(action, BOARD_SIZE)

            if p_shots_taken[r, c] != 0:
                turn = "human"  # Bezpiecznik
            else:
                ai_moves += 1
                total_turns += 1

                if p_board[r, c] > 0:
                    sid = int(p_board[r, c])
                    p_shots_taken[r, c] = 2
                    ai_view_on_p[r, c] = 2
                    p_hits[sid] += 1

                    if p_hits[sid] == len(p_cells[sid]):
                        msg = f"Przeciwnik zatopił nasz {len(p_cells[sid])}-masztowiec!"
                        mark_sunk_border_as_miss(ai_view_on_p, p_cells[sid])
                        audio.play_sfx("sunk")
                        add_anim(r, c, left_x, y0, "sunk")
                    else:
                        msg = "Dostaliśmy kulą armatnią! Okręt płonie!"
                        audio.play_sfx("hit")
                        add_anim(r, c, left_x, y0, "hit")

                    ai_ready_time = now_ms + AI_COOLDOWN_MS
                else:
                    p_shots_taken[r, c] = 1
                    ai_view_on_p[r, c] = 1
                    msg = "Przeciwnik chybił. Kapitan dowodzi atakiem!"
                    audio.play_sfx("miss")
                    add_anim(r, c, left_x, y0, "miss")
                    turn = "human"

                if all_sunk(p_cells, p_hits):
                    game_over = True
                    msg = "PORAŻKA! Nasza flota spoczywa na dnie."
                    audio.stop_music()
                    audio.play_sfx("lose")

        # =========================
        # RYSOWANIE
        # =========================
        draw_wave_bg(screen)

        # Top bar
        pygame.draw.rect(screen, (67, 52, 39), (0, 0, WIDTH, 40))
        title_txt = "Bitwa Morska: Piraci z Karaibów"
        screen.blit(big.render(title_txt, True, (242, 231, 211)), (MARGIN, 2))

        if phase == "setup":
            title = big.render("Wybór Ustawienia Floty", True, (30, 30, 30))
            screen.blit(title, (MARGIN, 55))

            if p_board is not None:
                tmp = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
                draw_board(screen, p_board, tmp, p_cells, p_hits, MARGIN, y0, show_ships=True)
                draw_grid_labels(screen, MARGIN, y0, small)
            else:
                hint = font.render("Brak statków na wodzie. Wylosuj układ.", True, (35, 35, 35))
                screen.blit(hint, (MARGIN, y0 + 30))

            draw_button(screen, btn_random, "Wylosuj układ", font, bg=(74, 126, 175))
            draw_button(screen, btn_start, "Wypłyń w rejs", font, bg=(46, 139, 87))
            draw_button(screen, btn_new_pool, "Nowa pula", font, bg=(120, 120, 120))

            music_label = "Dźwięk: WŁ" if audio.enabled else "Dźwięk: WYŁ"
            draw_button(screen, btn_music_setup, music_label, font, bg=(125, 88, 54))

            for i, line in enumerate(wrap_text(msg, max_chars=100)[:3]):
                screen.blit(small.render(line, True, (10, 10, 10)), (MARGIN, btn_random.bottom + 20 + i * 20))

        else:
            draw_board(screen, p_board, p_shots_taken, p_cells, p_hits, left_x, y0, show_ships=True)
            draw_board(screen, ai_board, ai_shots_taken, ai_cells, ai_hits, right_x, y0, show_ships=False)

            draw_grid_labels(screen, left_x, y0, small)
            draw_grid_labels(screen, right_x, y0, small)

            screen.blit(font.render("Flota Kapitana", True, (20, 20, 20)), (left_x, y0 - 65))
            screen.blit(font.render("Flota Wroga (AI)", True, (20, 20, 20)), (right_x, y0 - 65))

            # Animacje cząsteczek nad planszą
            update_and_draw_anims(screen)

            # Panel Dolny
            panel = pygame.Rect(MARGIN, status_y, WIDTH - 2 * MARGIN, STATUS_H - 8)
            pygame.draw.rect(screen, (235, 235, 235), panel, border_radius=10)
            pygame.draw.rect(screen, (118, 118, 118), panel, 1, border_radius=10)

            counters = f"Ruchy gracza: {human_moves}    Ruchy wroga: {ai_moves}    Łącznie ruchów: {total_turns}"
            screen.blit(small.render(counters, True, (30, 30, 30)), (MARGIN + 15, status_y + 15))

            turn_txt = "Teraz strzela: KAPITAN" if turn == "human" else "Teraz strzela: WRÓG (AI)"
            if game_over:
                turn_txt = "KONIEC BITWY"

            legend = "Legenda: Szare = Pudło  |  Czerwony X = Trafienie  |  Czerwone pole = Zatopiony"
            screen.blit(font.render(turn_txt, True, (20, 20, 180) if turn == "human" else (180, 20, 20)),
                        (MARGIN + 15, status_y + 45))
            screen.blit(small.render(legend, True, (60, 60, 60)), (MARGIN + 15, status_y + 75))

            for i, line in enumerate(wrap_text(msg, max_chars=100)[:3]):
                screen.blit(small.render(line, True, (10, 10, 10)), (MARGIN + 15, status_y + 105 + i * 18))

            music_label = "Dźwięk: WŁ" if audio.enabled else "Dźwięk: WYŁ"
            draw_button(screen, btn_music_game, music_label, small, bg=(125, 88, 54))

            # Ekran końca gry z półprzezroczystym tłem (overlay)
            if game_over:
                overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 200))
                screen.blit(overlay, (0, 0))

                end_text = "ZWYCIĘSTWO KAPITANIE!" if all_sunk(ai_cells, ai_hits) else "PORAŻKA... FLOTA ZATOPIONA"
                color = (60, 220, 80) if all_sunk(ai_cells, ai_hits) else (220, 60, 60)

                t = huge.render(end_text, True, color)
                screen.blit(t, (WIDTH // 2 - t.get_width() // 2, HEIGHT // 2 - 80))

                draw_button(screen, btn_restart, "Zagraj ponownie", big, bg=(150, 110, 50))

        pygame.display.flip()
        clock.tick(30)


if __name__ == "__main__":
    main()