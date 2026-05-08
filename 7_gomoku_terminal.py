#!/usr/bin/env python3
"""Simple two-player terminal Gomoku (Five-in-a-Row)."""

from __future__ import annotations

from dataclasses import dataclass


BOARD_SIZE = 15
EMPTY = "."


@dataclass
class Gomoku:
    size: int = BOARD_SIZE

    def __post_init__(self) -> None:
        self.board = [[EMPTY for _ in range(self.size)] for _ in range(self.size)]
        self.current = "X"
        self.moves = 0

    def print_board(self) -> None:
        header = "   " + " ".join(f"{i:2d}" for i in range(self.size))
        print(header)
        for r in range(self.size):
            print(f"{r:2d} " + " ".join(f" {c}" for c in self.board[r]))

    def place(self, row: int, col: int) -> bool:
        if not (0 <= row < self.size and 0 <= col < self.size):
            print("坐标越界，请输入 0-14 的行列号。")
            return False
        if self.board[row][col] != EMPTY:
            print("该位置已有棋子，请重新输入。")
            return False
        self.board[row][col] = self.current
        self.moves += 1
        return True

    def has_five(self, row: int, col: int) -> bool:
        piece = self.board[row][col]
        directions = ((1, 0), (0, 1), (1, 1), (1, -1))
        for dr, dc in directions:
            count = 1
            count += self._count_one_side(row, col, dr, dc, piece)
            count += self._count_one_side(row, col, -dr, -dc, piece)
            if count >= 5:
                return True
        return False

    def _count_one_side(self, row: int, col: int, dr: int, dc: int, piece: str) -> int:
        count = 0
        r, c = row + dr, col + dc
        while 0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == piece:
            count += 1
            r += dr
            c += dc
        return count

    def switch_player(self) -> None:
        self.current = "O" if self.current == "X" else "X"


def parse_move(text: str) -> tuple[int, int] | None:
    parts = text.strip().split()
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def main() -> None:
    game = Gomoku()
    print("五子棋开始！玩家 X 先手，输入格式：行 列，例如：7 7")

    while True:
        game.print_board()
        move_text = input(f"玩家 {game.current} 落子: ").strip()

        if move_text.lower() in {"q", "quit", "exit"}:
            print("游戏结束。")
            break

        move = parse_move(move_text)
        if move is None:
            print("输入格式错误，请输入两个整数，例如：7 7")
            continue

        row, col = move
        if not game.place(row, col):
            continue

        if game.has_five(row, col):
            game.print_board()
            print(f"玩家 {game.current} 获胜！")
            break

        if game.moves == game.size * game.size:
            game.print_board()
            print("棋盘已满，平局！")
            break

        game.switch_player()


if __name__ == "__main__":
    main()
