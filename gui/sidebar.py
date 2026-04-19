import pygame
from go_engine.board import BLACK, WHITE

BG = (45, 40, 35)
FG = (220, 215, 200)
ACC = (180, 130, 60)

class Sidebar:
    def __init__(self, surface, x, width, height):
        self.surface = surface
        self.x = x; self.w = width; self.h = height
        self._fl = self._fs = None

    def _fonts(self):
        if self._fl is None:
            self._fl = pygame.font.SysFont("Arial", 18, bold=True)
            self._fs = pygame.font.SysFont("Arial", 14)
        return self._fl, self._fs

    def draw(self, current_player, captured, ai_thinking, bs=0.0, ws=0.0):
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
