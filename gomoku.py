"""
@ Author: Peter Xiao / extended
@ Date: 2026.04.23
@ Filename: gomoku.py
@ Brief: 五子棋小游戏（人人对战 & 人机对战）
         使用 Python 内置 tkinter，无需额外依赖
         运行: python gomoku.py
"""

import tkinter as tk
from tkinter import messagebox

# ──────────────────────────────────────────────
# 游戏配置
# ──────────────────────────────────────────────
BOARD_SIZE   = 15        # 棋盘格数 (15×15)
CELL_SIZE    = 40        # 每格像素
MARGIN       = 30        # 边距
STONE_RADIUS = 17        # 棋子半径

BOARD_PX = MARGIN * 2 + CELL_SIZE * (BOARD_SIZE - 1)

BLACK = 1
WHITE = 2
EMPTY = 0

COLOR_BG    = "#DEB887"
COLOR_LINE  = "#8B5E3C"
COLOR_STAR  = "#5C3317"
COLOR_BLACK = "#111111"
COLOR_WHITE = "#F5F5F5"
COLOR_HINT  = "#FF4444"

# 天元及星位坐标（15路棋盘）
STAR_POINTS = [(3,3),(3,11),(11,3),(11,11),(7,7),
               (3,7),(7,3),(11,7),(7,11)]


# ──────────────────────────────────────────────
# 棋盘逻辑
# ──────────────────────────────────────────────
class Board:
    def __init__(self):
        self.grid = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.history = []        # [(row, col, player)]

    def reset(self):
        self.grid = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        self.history.clear()

    def place(self, row, col, player):
        self.grid[row][col] = player
        self.history.append((row, col, player))

    def undo(self):
        if not self.history:
            return None
        row, col, player = self.history.pop()
        self.grid[row][col] = EMPTY
        return row, col, player

    def is_empty(self, row, col):
        return self.grid[row][col] == EMPTY

    def in_bounds(self, row, col):
        return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE

    def check_win(self, row, col, player):
        """检查 (row, col) 处落子后 player 是否五连"""
        directions = [(0,1),(1,0),(1,1),(1,-1)]
        for dr, dc in directions:
            count = 1
            for sign in (1, -1):
                r, c = row + sign*dr, col + sign*dc
                while self.in_bounds(r, c) and self.grid[r][c] == player:
                    count += 1
                    r += sign*dr
                    c += sign*dc
            if count >= 5:
                return True
        return False

    def is_full(self):
        return all(self.grid[r][c] != EMPTY
                   for r in range(BOARD_SIZE)
                   for c in range(BOARD_SIZE))


# ──────────────────────────────────────────────
# AI（启发式评分）
# ──────────────────────────────────────────────
SCORE_TABLE = {
    (5, True):  1_000_000,   # 五连
    (4, True):    50_000,    # 活四
    (4, False):   10_000,    # 冲四
    (3, True):     5_000,    # 活三
    (3, False):      500,    # 眠三
    (2, True):       200,    # 活二
    (2, False):       50,    # 眠二
    (1, True):        10,
    (1, False):        2,
}


def _line_score(grid, row, col, player, dr, dc):
    """计算一个方向上以 (row,col) 为中心的连线得分"""
    count = 1
    blocked_left = blocked_right = False

    r, c = row + dr, col + dc
    while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and grid[r][c] == player:
        count += 1; r += dr; c += dc
    if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
        blocked_right = True
    elif grid[r][c] != EMPTY:
        blocked_right = True

    r, c = row - dr, col - dc
    while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and grid[r][c] == player:
        count += 1; r -= dr; c -= dc
    if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
        blocked_left = True
    elif grid[r][c] != EMPTY:
        blocked_left = True

    if count >= 5:
        return SCORE_TABLE[(5, True)]   # 五连必胜，无论是否被堵
    alive = not (blocked_left or blocked_right)
    return SCORE_TABLE.get((count, alive), SCORE_TABLE.get((count, False), 0))


def evaluate_point(grid, row, col, player):
    """评估在 (row,col) 落子对 player 的价值"""
    directions = [(0,1),(1,0),(1,1),(1,-1)]
    return sum(_line_score(grid, row, col, player, *d) for d in directions)


def ai_move(board: Board, ai_player: int):
    """简单启发式 AI：选攻守综合最优点"""
    human = WHITE if ai_player == BLACK else BLACK
    best_score = -1
    best_pos   = None

    candidates = _candidate_cells(board)
    if not candidates:
        # 棋盘为空，走中心
        return BOARD_SIZE // 2, BOARD_SIZE // 2

    for row, col in candidates:
        # 进攻得分
        board.grid[row][col] = ai_player
        atk = evaluate_point(board.grid, row, col, ai_player)
        board.grid[row][col] = EMPTY

        # 防守得分（模拟对手落在此处）
        board.grid[row][col] = human
        dfs = evaluate_point(board.grid, row, col, human)
        board.grid[row][col] = EMPTY

        score = max(atk, dfs * 0.9)
        if score > best_score:
            best_score = score
            best_pos   = (row, col)

    return best_pos


def _candidate_cells(board: Board):
    """只考虑已有棋子周围 2 格范围内的空点"""
    visited = set()
    cands   = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board.grid[r][c] != EMPTY:
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        nr, nc = r+dr, c+dc
                        if (board.in_bounds(nr, nc) and
                                board.is_empty(nr, nc) and
                                (nr, nc) not in visited):
                            visited.add((nr, nc))
                            cands.append((nr, nc))
    return cands


# ──────────────────────────────────────────────
# GUI
# ──────────────────────────────────────────────
class GomokuApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("五子棋 Gomoku")
        self.root.resizable(False, False)

        self.board  = Board()
        self.mode   = tk.StringVar(value="pvp")   # "pvp" | "pvc"
        self.ai_side = tk.StringVar(value="white") # "black" | "white"
        self.current_player = BLACK
        self.game_over = False
        self.last_pos  = None    # 最后落子坐标，用于高亮

        self._build_ui()
        self._new_game()

    # ── UI 构建 ──────────────────────────────
    def _build_ui(self):
        ctrl = tk.Frame(self.root, bg="#3C2A1E", padx=8, pady=6)
        ctrl.pack(fill=tk.X)

        tk.Label(ctrl, text="五子棋", fg="#FFD700", bg="#3C2A1E",
                 font=("Helvetica", 16, "bold")).pack(side=tk.LEFT, padx=8)

        # 模式选择
        mode_frame = tk.LabelFrame(ctrl, text="模式", fg="#DEB887",
                                   bg="#3C2A1E", font=("Helvetica", 9))
        mode_frame.pack(side=tk.LEFT, padx=6)
        for text, val in [("人人对战", "pvp"), ("人机对战", "pvc")]:
            tk.Radiobutton(mode_frame, text=text, variable=self.mode, value=val,
                           bg="#3C2A1E", fg="#DEB887", selectcolor="#5C3317",
                           activebackground="#3C2A1E",
                           command=self._new_game).pack(side=tk.LEFT, padx=3)

        # AI 执子选择
        ai_frame = tk.LabelFrame(ctrl, text="AI执", fg="#DEB887",
                                  bg="#3C2A1E", font=("Helvetica", 9))
        ai_frame.pack(side=tk.LEFT, padx=6)
        for text, val in [("黑", "black"), ("白", "white")]:
            tk.Radiobutton(ai_frame, text=text, variable=self.ai_side, value=val,
                           bg="#3C2A1E", fg="#DEB887", selectcolor="#5C3317",
                           activebackground="#3C2A1E",
                           command=self._on_ai_side_change).pack(side=tk.LEFT, padx=3)

        # 按钮
        btn_cfg = dict(bg="#5C3317", fg="#FFD700", font=("Helvetica", 10, "bold"),
                       relief=tk.FLAT, padx=10, pady=2, cursor="hand2")
        tk.Button(ctrl, text="新局", command=self._new_game,
                  **btn_cfg).pack(side=tk.RIGHT, padx=4)
        tk.Button(ctrl, text="悔棋", command=self._undo,
                  **btn_cfg).pack(side=tk.RIGHT, padx=4)

        # 状态栏
        self.status_var = tk.StringVar()
        tk.Label(self.root, textvariable=self.status_var,
                 bg="#2C1A0E", fg="#FFD700",
                 font=("Helvetica", 11), pady=3).pack(fill=tk.X)

        # 棋盘 Canvas
        self.canvas = tk.Canvas(self.root, width=BOARD_PX, height=BOARD_PX,
                                bg=COLOR_BG, highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Motion>",   self._on_hover)
        self._hover_pos = None

    # ── 坐标转换 ──────────────────────────────
    def _pixel_to_cell(self, x, y):
        col = round((x - MARGIN) / CELL_SIZE)
        row = round((y - MARGIN) / CELL_SIZE)
        return row, col

    def _cell_to_pixel(self, row, col):
        return MARGIN + col * CELL_SIZE, MARGIN + row * CELL_SIZE

    # ── 绘制 ──────────────────────────────────
    def _draw_board(self):
        self.canvas.delete("all")

        # 棋盘线
        for i in range(BOARD_SIZE):
            x0 = MARGIN + i * CELL_SIZE
            self.canvas.create_line(x0, MARGIN, x0,
                                    MARGIN + (BOARD_SIZE-1)*CELL_SIZE,
                                    fill=COLOR_LINE, width=1)
            self.canvas.create_line(MARGIN, x0,
                                    MARGIN + (BOARD_SIZE-1)*CELL_SIZE, x0,
                                    fill=COLOR_LINE, width=1)

        # 边框加粗
        m = MARGIN; e = MARGIN + (BOARD_SIZE-1)*CELL_SIZE
        self.canvas.create_rectangle(m, m, e, e, outline=COLOR_LINE, width=2)

        # 星位
        for r, c in STAR_POINTS:
            px, py = self._cell_to_pixel(r, c)
            self.canvas.create_oval(px-4, py-4, px+4, py+4,
                                    fill=COLOR_STAR, outline="")

        # 棋子
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                stone = self.board.grid[r][c]
                if stone != EMPTY:
                    self._draw_stone(r, c, stone,
                                     highlight=(r, c) == self.last_pos)

        # 悬停提示
        if self._hover_pos and not self.game_over:
            r, c = self._hover_pos
            if self.board.is_empty(r, c):
                px, py = self._cell_to_pixel(r, c)
                color = COLOR_BLACK if self.current_player == BLACK else COLOR_WHITE
                self.canvas.create_oval(px-STONE_RADIUS, py-STONE_RADIUS,
                                        px+STONE_RADIUS, py+STONE_RADIUS,
                                        fill=color, outline="", stipple="gray50")

    def _draw_stone(self, row, col, player, highlight=False):
        px, py = self._cell_to_pixel(row, col)
        r = STONE_RADIUS
        if player == BLACK:
            self.canvas.create_oval(px-r, py-r, px+r, py+r,
                                    fill=COLOR_BLACK, outline="#444")
            if highlight:
                self.canvas.create_oval(px-5, py-5, px+5, py+5,
                                        fill=COLOR_HINT, outline="")
        else:
            self.canvas.create_oval(px-r, py-r, px+r, py+r,
                                    fill=COLOR_WHITE, outline="#AAA")
            if highlight:
                self.canvas.create_oval(px-5, py-5, px+5, py+5,
                                        fill=COLOR_HINT, outline="")

    # ── 游戏逻辑 ──────────────────────────────
    def _new_game(self):
        self.board.reset()
        self.current_player = BLACK
        self.game_over      = False
        self.last_pos       = None
        self._hover_pos     = None
        self._update_status()
        self._draw_board()
        # 人机模式且 AI 执黑，AI 先手
        if self.mode.get() == "pvc" and self._ai_player() == BLACK:
            self.root.after(300, self._ai_turn)

    def _ai_player(self):
        return BLACK if self.ai_side.get() == "black" else WHITE

    def _is_human_turn(self):
        if self.mode.get() == "pvp":
            return True
        return self.current_player != self._ai_player()

    def _on_ai_side_change(self):
        if self.mode.get() == "pvc":
            self._new_game()

    def _on_hover(self, event):
        row, col = self._pixel_to_cell(event.x, event.y)
        if self.board.in_bounds(row, col):
            if (row, col) != self._hover_pos:
                self._hover_pos = (row, col)
                self._draw_board()
        else:
            if self._hover_pos is not None:
                self._hover_pos = None
                self._draw_board()

    def _on_click(self, event):
        if self.game_over or not self._is_human_turn():
            return
        row, col = self._pixel_to_cell(event.x, event.y)
        if not self.board.in_bounds(row, col):
            return
        if not self.board.is_empty(row, col):
            return
        self._do_place(row, col)

    def _do_place(self, row, col):
        self.board.place(row, col, self.current_player)
        self.last_pos = (row, col)
        self._draw_board()

        if self.board.check_win(row, col, self.current_player):
            winner = "黑棋" if self.current_player == BLACK else "白棋"
            self.game_over = True
            self.status_var.set(f"{winner} 获胜！")
            messagebox.showinfo("游戏结束", f"{winner} 获胜！")
            return

        if self.board.is_full():
            self.game_over = True
            self.status_var.set("平局！")
            messagebox.showinfo("游戏结束", "平局！")
            return

        self.current_player = WHITE if self.current_player == BLACK else BLACK
        self._update_status()

        # 切换到 AI 回合
        if self.mode.get() == "pvc" and not self._is_human_turn():
            self.root.after(200, self._ai_turn)

    def _ai_turn(self):
        if self.game_over:
            return
        pos = ai_move(self.board, self._ai_player())
        if pos:
            self._do_place(*pos)

    def _undo(self):
        if not self.board.history:
            return

        # 人机模式下撤销两步（AI 的也一起撤），且支持终局后悔棋。
        self.game_over = False
        steps = 2 if self.mode.get() == "pvc" else 1
        for _ in range(steps):
            result = self.board.undo()
            if result is None:
                break

        if self.board.history:
            self.last_pos = (self.board.history[-1][0], self.board.history[-1][1])
        else:
            self.last_pos = None

        # 按棋谱长度恢复当前执子：黑先，偶数步轮到黑，奇数步轮到白。
        self.current_player = BLACK if len(self.board.history) % 2 == 0 else WHITE
        self._update_status()
        self._draw_board()

    def _update_status(self):
        player_name = "黑棋" if self.current_player == BLACK else "白棋"
        if self.mode.get() == "pvc" and not self._is_human_turn():
            player_name += "（AI）"
        self.status_var.set(f"当前回合：{player_name}  ●=黑  ○=白")


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = GomokuApp(root)
    root.mainloop()
