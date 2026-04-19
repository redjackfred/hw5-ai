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

    def do_ai_move():
        nonlocal thinking, last_move
        if not engine.game.is_over() and engine.game.get_legal_moves():
            move = ai.select_move(engine.game, temperature=0.0)
            engine.place_stone(*move)
            last_move = move
        thinking = False

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            if not thinking and not engine.game.is_over():
                human = (engine.game.current_player == BLACK) and not args.ai_vs_ai
                if human:
                    click = ih.get_board_click(ev)
                    if click and engine.is_legal(*click):
                        engine.place_stone(*click)
                        last_move = click
                        if ai:
                            thinking = True
                            threading.Thread(target=do_ai_move, daemon=True).start()

        if args.ai_vs_ai and not thinking and not engine.game.is_over():
            thinking = True
            threading.Thread(target=do_ai_move, daemon=True).start()

        screen.fill((30, 25, 20))
        bv.draw(engine.get_board(), last_move)
        bs, ws = engine.get_score()
        sb.draw(engine.game.current_player, engine.game.captured, thinking, bs, ws)

        if engine.game.is_over() or not engine.game.get_legal_moves():
            w = engine.get_winner()
            f = pygame.font.SysFont("Arial", 32, bold=True)
            s = f.render(f"{w.capitalize()} wins!", True, (255, 220, 80))
            screen.blit(s, (BOARD_PX//2 - s.get_width()//2, H//2))

        pygame.display.flip()
        clock.tick(30)
    pygame.quit()


if __name__ == "__main__":
    main()
