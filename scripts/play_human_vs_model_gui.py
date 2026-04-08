import pygame
import numpy as np
import torch
import torch.nn as nn

from envs.battleship_rl_env import BOARD_SIZE, FLEET

CELL = 36
MARGIN = 30
BOARD_PIX = BOARD_SIZE * CELL
STATUS_H = 120
WIDTH = 2 * BOARD_PIX + 3 * MARGIN
HEIGHT = BOARD_PIX + 3 * MARGIN + STATUS_H


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


def model_action(model, obs, legal_mask):
    legal = np.where(legal_mask == 1)[0]
    if len(legal) == 0:
        return 0
    with torch.no_grad():
        s = torch.tensor(obs[None, None, :, :], dtype=torch.float32)
        q = model(s).numpy()[0]
    q_masked = np.full_like(q, -1e9, dtype=np.float32)
    q_masked[legal] = q[legal]
    return int(np.argmax(q_masked))


def draw_x(screen, rect, color=(200, 0, 0), thickness=3):
    pygame.draw.line(screen, color, (rect.left + 5, rect.top + 5), (rect.right - 5, rect.bottom - 5), thickness)
    pygame.draw.line(screen, color, (rect.left + 5, rect.bottom - 5), (rect.right - 5, rect.top + 5), thickness)


def draw_grid_labels(screen, x0, y0, font):
    cols = "ABCDEFGHIJ"
    for c in range(BOARD_SIZE):
        txt = font.render(cols[c], True, (20, 20, 20))
        tx = x0 + c * CELL + CELL // 2 - txt.get_width() // 2
        ty = y0 - 20
        screen.blit(txt, (tx, ty))
    for r in range(BOARD_SIZE):
        txt = font.render(str(r + 1), True, (20, 20, 20))
        tx = x0 - 22
        ty = y0 + r * CELL + CELL // 2 - txt.get_height() // 2
        screen.blit(txt, (tx, ty))


def draw_board(screen, ships_board, shots_taken, ship_cells, ship_hits, x0, y0, show_ships=False):
    WATER = (220, 235, 250)
    MISS = (170, 170, 170)
    SHIP = (80, 140, 220)
    SUNK = (220, 60, 60)
    HIT_BG = (245, 245, 245)

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            rect = pygame.Rect(x0 + c * CELL, y0 + r * CELL, CELL, CELL)
            ship_id = int(ships_board[r, c])
            shot = int(shots_taken[r, c])

            color = WATER
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
                color = SHIP if (ship_id > 0 and show_ships) else WATER

            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (40, 40, 40), rect, 1)

            if draw_hit_x:
                draw_x(screen, rect, color=(200, 0, 0), thickness=3)


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


def draw_button(screen, rect, text, font, bg=(70, 130, 180), fg=(255, 255, 255)):
    pygame.draw.rect(screen, bg, rect, border_radius=8)
    pygame.draw.rect(screen, (30, 30, 30), rect, 2, border_radius=8)
    t = font.render(text, True, fg)
    screen.blit(t, (rect.x + rect.w // 2 - t.get_width() // 2, rect.y + rect.h // 2 - t.get_height() // 2))


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Statki: Gracz vs Model (CNN)")
    font = pygame.font.SysFont("arial", 20)
    small = pygame.font.SysFont("arial", 16)
    big = pygame.font.SysFont("arial", 28)
    clock = pygame.time.Clock()

    model = QNetCNN()
    model.load_state_dict(torch.load("checkpoints/selfplay_best.pt", map_location="cpu"))
    model.eval()

    rng = np.random.default_rng()
    seen_layouts = set()

    p_board = p_cells = p_hits = None
    ai_board = ai_cells = ai_hits = None
    p_shots_taken = ai_shots_taken = None
    p_view_on_ai = ai_view_on_p = None

    turn = "human"
    game_over = False
    msg = "Kliknij 'Wylosuj ustawienie', potem 'Start gry'."
    phase = "setup"

    left_x = MARGIN
    right_x = 2 * MARGIN + BOARD_PIX
    y0 = MARGIN + 25

    btn_random = pygame.Rect(MARGIN, y0 + BOARD_PIX + 20, 240, 48)
    btn_start = pygame.Rect(MARGIN + 260, y0 + BOARD_PIX + 20, 180, 48)
    btn_new_pool = pygame.Rect(MARGIN + 460, y0 + BOARD_PIX + 20, 180, 48)

    def start_match():
        nonlocal ai_board, ai_cells, ai_hits
        nonlocal p_shots_taken, ai_shots_taken, p_view_on_ai, ai_view_on_p
        nonlocal turn, game_over
        ai_board, ai_cells, ai_hits = random_place_fleet(rng)
        p_shots_taken = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        ai_shots_taken = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        p_view_on_ai = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        ai_view_on_p = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        turn = "human"
        game_over = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            if phase == "setup":
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if btn_random.collidepoint(event.pos):
                        p_board, p_cells, p_hits, repeated = generate_unique_player_board(rng, seen_layouts)
                        if repeated:
                            msg = "Wylosowano ustawienie (po wielu losowaniach mogła wystąpić powtórka)."
                        else:
                            msg = "Wylosowano nowe ustawienie statków."
                    elif btn_start.collidepoint(event.pos):
                        if p_board is None:
                            msg = "Najpierw kliknij 'Wylosuj ustawienie'."
                        else:
                            start_match()
                            phase = "game"
                            msg = "Start! Twoja tura."
                    elif btn_new_pool.collidepoint(event.pos):
                        seen_layouts.clear()
                        p_board = p_cells = p_hits = None
                        msg = "Wyczyszczono pulę losowań. Wylosuj ustawienie."

            elif phase == "game":
                if not game_over and turn == "human" and event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = event.pos
                    if right_x <= mx < right_x + BOARD_PIX and y0 <= my < y0 + BOARD_PIX:
                        c = (mx - right_x) // CELL
                        r = (my - y0) // CELL

                        if ai_shots_taken[r, c] != 0:
                            msg = "To pole było już ostrzelane. Wybierz inne."
                        else:
                            if ai_board[r, c] > 0:
                                sid = int(ai_board[r, c])
                                ai_shots_taken[r, c] = 2
                                p_view_on_ai[r, c] = 2
                                ai_hits[sid] += 1
                                if ai_hits[sid] == len(ai_cells[sid]):
                                    msg = f"TRAFIONY I ZATOPIONY ({len(ai_cells[sid])}-masztowy). Grasz dalej!"
                                else:
                                    msg = "TRAFIONY! Grasz dalej."
                            else:
                                ai_shots_taken[r, c] = 1
                                p_view_on_ai[r, c] = 1
                                msg = "PUDŁO. Teraz ruch AI."
                                turn = "ai"

                            if all_sunk(ai_cells, ai_hits):
                                game_over = True
                                msg = "WYGRAŁEŚ! Zatopiłeś wszystkie statki przeciwnika."

        if phase == "game" and not game_over and turn == "ai":
            obs = ai_view_on_p.astype(np.float32)
            legal_mask = (ai_view_on_p.reshape(-1) == 0).astype(np.int8)
            action = model_action(model, obs, legal_mask)
            r, c = divmod(action, BOARD_SIZE)

            if p_shots_taken[r, c] != 0:
                msg = "AI wybrało pole już ostrzelane. Twoja tura."
                turn = "human"
            else:
                if p_board[r, c] > 0:
                    sid = int(p_board[r, c])
                    p_shots_taken[r, c] = 2
                    ai_view_on_p[r, c] = 2
                    p_hits[sid] += 1
                    if p_hits[sid] == len(p_cells[sid]):
                        msg = f"AI: TRAFIONY I ZATOPIONY ({len(p_cells[sid])}-masztowy). AI strzela ponownie."
                    else:
                        msg = "AI: TRAFIONY. AI strzela ponownie."
                else:
                    p_shots_taken[r, c] = 1
                    ai_view_on_p[r, c] = 1
                    msg = "AI: PUDŁO. Twoja tura."
                    turn = "human"

                if all_sunk(p_cells, p_hits):
                    game_over = True
                    msg = "AI WYGRYWA. Zatopiło wszystkie Twoje statki."

        screen.fill((245, 245, 245))

        if phase == "setup":
            title = big.render("Statki: losowanie ustawienia", True, (20, 20, 20))
            screen.blit(title, (MARGIN, 8))

            preview_y = MARGIN + 45
            if p_board is not None:
                tmp_shots = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
                draw_board(screen, p_board, tmp_shots, p_cells, p_hits, MARGIN, preview_y, show_ships=True)
                draw_grid_labels(screen, MARGIN, preview_y, small)
            else:
                hint = font.render("Brak ustawienia. Kliknij 'Wylosuj ustawienie'.", True, (30, 30, 30))
                screen.blit(hint, (MARGIN, preview_y + 20))

            draw_button(screen, btn_random, "Wylosuj ustawienie", font)
            draw_button(screen, btn_start, "Start gry", font, bg=(46, 139, 87))
            draw_button(screen, btn_new_pool, "Nowa pula", font, bg=(120, 120, 120))

            info = "Losowanie stara się unikać powtórek układu."
            screen.blit(small.render(info, True, (30, 30, 30)), (MARGIN, btn_random.y + 60))

            for i, line in enumerate(wrap_text(msg, max_chars=95)[:3]):
                screen.blit(small.render(line, True, (10, 10, 10)), (MARGIN, btn_random.y + 85 + i * 20))

        else:
            draw_board(screen, p_board, p_shots_taken, p_cells, p_hits, left_x, y0, show_ships=True)
            draw_board(screen, ai_board, ai_shots_taken, ai_cells, ai_hits, right_x, y0, show_ships=False)

            draw_grid_labels(screen, left_x, y0, small)
            draw_grid_labels(screen, right_x, y0, small)

            screen.blit(font.render("Moje statki", True, (20, 20, 20)), (left_x, 5))
            screen.blit(font.render("Przeciwnik", True, (20, 20, 20)), (right_x, 5))

            status_y = y0 + BOARD_PIX + 15
            panel_rect = pygame.Rect(MARGIN, status_y, WIDTH - 2 * MARGIN, STATUS_H - 10)
            pygame.draw.rect(screen, (235, 235, 235), panel_rect)
            pygame.draw.rect(screen, (120, 120, 120), panel_rect, 1)

            legend = "Legenda: pudło = szare, trafiony = czerwony X, trafiony i zatopiony = czerwone pole."
            turn_txt = "Tura: GRACZ" if turn == "human" else "Tura: AI"
            if game_over:
                turn_txt = "KONIEC GRY"

            screen.blit(small.render(legend, True, (30, 30, 30)), (MARGIN + 8, status_y + 8))
            screen.blit(small.render(turn_txt, True, (30, 30, 30)), (MARGIN + 8, status_y + 30))

            for i, line in enumerate(wrap_text(msg, max_chars=95)[:3]):
                screen.blit(small.render(line, True, (10, 10, 10)), (MARGIN + 8, status_y + 52 + i * 20))

        pygame.display.flip()
        clock.tick(30)


if __name__ == "__main__":
    main()