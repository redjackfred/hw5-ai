"""
python main.py                            # Human (Black) vs AI (White)
python main.py --ai-vs-ai                 # AI vs AI demo
python main.py --checkpoint path/to.pt   # custom checkpoint
"""
import argparse, os, threading
import torch, pygame
from go_engine.board import BLACK, WHITE
from go_engine.scorer import compute_score
from api.engine_api import GoEngine
from gui.board_view import BoardView, BOARD_PX
from gui.sidebar import Sidebar
from gui.input_handler import InputHandler
from model.network import GoNetwork
from mcts.mcts import MCTS

SIDEBAR_W = 160
W, H = BOARD_PX + SIDEBAR_W, BOARD_PX
AI_RESIGN_THRESHOLD = -0.6  # Q < -0.6 → ~20% win prob → AI resigns


def load_ai(ckpt):
    if not os.path.exists(ckpt):
        print(f"No checkpoint at {ckpt}. AI plays with untrained network.")
    device = GoNetwork.get_device()
    net = GoNetwork().to(device)
    if os.path.exists(ckpt):
        net.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    net.eval()
    return MCTS(net, num_simulations=1600, time_limit=3.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/rl_best.pt")
    ap.add_argument("--ai-vs-ai", action="store_true")
    args = ap.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("9x9 Go — AlphaGo")
    clock = pygame.time.Clock()

    bv = BoardView(screen)
    sb = Sidebar(screen, BOARD_PX, SIDEBAR_W, H)
    ih = InputHandler()
    engine = GoEngine(); engine.new_game()
    ai = load_ai(args.checkpoint)
    engine.set_ai(ai)

    thinking = False
    last_move = None
    resigned = None  # 'black' or 'white' — whoever resigned

    def do_ai_move():
        nonlocal thinking, last_move, resigned
        if not engine.game.is_over() and engine.game.get_legal_moves():
            move = ai.select_move(engine.game, temperature=0.0,
                                  resign_threshold=AI_RESIGN_THRESHOLD)
            if move is None:
                ai_color = engine.game.current_player
                resigned = 'black' if ai_color == BLACK else 'white'
            else:
                engine.place_stone(*move)
                last_move = move
        thinking = False

    running = True
    while running:
        resign_rect = None
        game_over = resigned is not None or engine.game.is_over() or not engine.game.get_legal_moves()
        is_human_turn = (engine.game.current_player == BLACK) and not args.ai_vs_ai

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

            if not thinking and not game_over:
                if is_human_turn:
                    # Human resign button
                    if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                        if sb._resign_rect and sb._resign_rect.collidepoint(ev.pos):
                            resigned = 'black'
                            continue
                    # Board click
                    click = ih.get_board_click(ev)
                    if click and engine.is_legal(*click):
                        engine.place_stone(*click)
                        last_move = click
                        thinking = True
                        threading.Thread(target=do_ai_move, daemon=True).start()

        if args.ai_vs_ai and not thinking and not game_over:
            thinking = True
            threading.Thread(target=do_ai_move, daemon=True).start()

        screen.fill((30, 25, 20))
        bv.draw(engine.get_board(), last_move)
        bs, ws = engine.get_score()
        show_resign = is_human_turn and not thinking and not game_over
        resign_rect = sb.draw(engine.game.current_player, engine.game.captured,
                              thinking, bs, ws, show_resign_btn=show_resign)

        if game_over:
            if resigned:
                winner = 'white' if resigned == 'black' else 'black'
                msg = f"{resigned.capitalize()} resigns!  {winner.capitalize()} wins!"
            else:
                msg = f"{engine.get_winner().capitalize()} wins!"
            f = pygame.font.SysFont("Arial", 28, bold=True)
            s = f.render(msg, True, (255, 220, 80))
            overlay = pygame.Surface((s.get_width() + 20, s.get_height() + 12), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 160))
            ox = BOARD_PX // 2 - overlay.get_width() // 2
            oy = H // 2 - overlay.get_height() // 2
            screen.blit(overlay, (ox, oy))
            screen.blit(s, (ox + 10, oy + 6))

        pygame.display.flip()
        clock.tick(30)
    pygame.quit()


if __name__ == "__main__":
    main()
