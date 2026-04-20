import pygame
from go_engine.board import BLACK, WHITE

BG = (45, 40, 35)
FG = (220, 215, 200)
ACC = (180, 130, 60)
BTN_BG = (140, 50, 50)
BTN_FG = (240, 220, 210)
BTN_HOVER = (180, 70, 70)

class Sidebar:
    def __init__(self, surface, x, width, height):
        self.surface = surface
        self.x = x; self.w = width; self.h = height
        self._fl = self._fs = None
        self._resign_rect = None

    def _fonts(self):
        if self._fl is None:
            self._fl = pygame.font.SysFont("Arial", 18, bold=True)
            self._fs = pygame.font.SysFont("Arial", 14)
        return self._fl, self._fs

    def draw(self, current_player, captured, ai_thinking, bs=0.0, ws=0.0,
             show_resign_btn=False) -> pygame.Rect | None:
        fl, fs = self._fonts()
        pygame.draw.rect(self.surface, BG, (self.x, 0, self.w, self.h))
        y = 30
        def t(msg, font, color=FG):
            nonlocal y
            s = font.render(msg, True, color)
            self.surface.blit(s, (self.x + 10, y))
            y += s.get_height() + 5
        t("Black", fl, (200,200,200))
        t(f"  Cap: {captured.get(BLACK,0)}", fs)
        t(f"  Score: {bs:.1f}", fs)
        y += 8
        t("White", fl, (240,240,240))
        t(f"  Cap: {captured.get(WHITE,0)}", fs)
        t(f"  Score: {ws:.1f}", fs)
        y += 12
        t(f"Turn: {'Black' if current_player==BLACK else 'White'}", fl, ACC)
        if ai_thinking:
            y += 8; t("AI thinking...", fs, (150,200,150))

        resign_rect = None
        if show_resign_btn:
            btn_y = self.h - 50
            btn_rect = pygame.Rect(self.x + 10, btn_y, self.w - 20, 32)
            mx, my = pygame.mouse.get_pos()
            hovered = btn_rect.collidepoint(mx, my)
            pygame.draw.rect(self.surface, BTN_HOVER if hovered else BTN_BG, btn_rect, border_radius=5)
            lbl = fl.render("Resign", True, BTN_FG)
            self.surface.blit(lbl, (btn_rect.centerx - lbl.get_width()//2,
                                    btn_rect.centery - lbl.get_height()//2))
            resign_rect = btn_rect

        self._resign_rect = resign_rect
        return resign_rect
