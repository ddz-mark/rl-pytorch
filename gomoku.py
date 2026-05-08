#!/usr/bin/env python3
"""Simple terminal Gomoku (Five in a Row) game."""

from __future__ import annotations

BOARD_SIZE = 15
EMPTY = "."
PLAYER_X = "X"
PLAYER_O = "O"


class GomokuGame:
    def __init__(self, size: int = BOARD_SIZE) -> None:
        self.size = size
        self.board = [[EMPTY for _ in range(size)] for _ in range(size)]
        self.current_player = PLAYER_X
        self.moves_played = 0

    def render(self) -> None:
        header = "   " + " ".join(f"{i + 1:2d}" for i in range(self.size))
        print(header)
        for row_idx, row in enumerate(self.board):
            row_label = f"{row_idx + 1:2d}"
            print(f"{row_label} " + " ".join(f" {cell}" for cell in row))

    def parse_move(self, raw: str) -> tuple[int, int] | None:
        cleaned = raw.strip().lower().replace(",", " ")
        if cleaned in {"q", "quit", "exit"}:
            return None

        parts = cleaned.split()
        if len(parts) != 2:
            raise ValueError("Please enter exactly two numbers like: 8 8")

        row_str, col_str = parts
        if not (row_str.isdigit() and col_str.isdigit()):
            raise ValueError("Row and column must both be numbers.")

        row, col = int(row_str) - 1, int(col_str) - 1
        if not (0 <= row < self.size and 0 <= col < self.size):
            raise ValueError(f"Move must be between 1 and {self.size}.")

        return row, col

    def place_stone(self, row: int, col: int) -> None:
        if self.board[row][col] != EMPTY:
            raise ValueError("That position is already occupied.")
        self.board[row][col] = self.current_player
        self.moves_played += 1

    def has_five_in_a_row(self, row: int, col: int) -> bool:
        stone = self.board[row][col]
        if stone == EMPTY:
            return False

        directions = (
            (1, 0),
            (0, 1),
            (1, 1),
            (1, -1),
        )

        for d_row, d_col in directions:
            count = 1
            count += self._count_direction(row, col, d_row, d_col, stone)
            count += self._count_direction(row, col, -d_row, -d_col, stone)
            if count >= 5:
                return True
        return False

    def _count_direction(self, row: int, col: int, d_row: int, d_col: int, stone: str) -> int:
        count = 0
        r, c = row + d_row, col + d_col
        while 0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == stone:
            count += 1
            r += d_row
            c += d_col
        return count

    def switch_player(self) -> None:
        self.current_player = PLAYER_O if self.current_player == PLAYER_X else PLAYER_X

    def run(self) -> None:
        print("Welcome to Gomoku (Five in a Row)!")
        print(f"Input format: row col (1-{self.size}), enter q to quit.")

        while True:
            print("\nCurrent board:")
            self.render()
            print(f"\nPlayer {self.current_player}, your move:")

            user_input = input("> ")
            try:
                parsed = self.parse_move(user_input)
            except ValueError as err:
                print(f"Invalid input: {err}")
                continue

            if parsed is None:
                print("Game ended by player request.")
                return

            row, col = parsed
            try:
                self.place_stone(row, col)
            except ValueError as err:
                print(f"Invalid move: {err}")
                continue

            if self.has_five_in_a_row(row, col):
                print("\nFinal board:")
                self.render()
                print(f"\nPlayer {self.current_player} wins!")
                return

            if self.moves_played == self.size * self.size:
                print("\nFinal board:")
                self.render()
                print("\nThe board is full. It's a draw!")
                return

            self.switch_player()


def main() -> None:
    GomokuGame().run()


if __name__ == "__main__":
    main()
