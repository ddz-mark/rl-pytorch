"""A simple command-line Gomoku (five-in-a-row) game."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


Board = List[List[str]]


@dataclass
class GomokuGame:
    """Core game state and rules for Gomoku."""

    size: int = 15
    win_length: int = 5
    board: Board = field(init=False)
    current_player: str = field(default="X", init=False)
    winner: Optional[str] = field(default=None, init=False)
    moves_played: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("Board size must be positive.")
        if self.win_length <= 1:
            raise ValueError("Win length must be greater than 1.")
        if self.win_length > self.size:
            raise ValueError("Win length cannot exceed board size.")

        self.board = [["." for _ in range(self.size)] for _ in range(self.size)]

    def place_stone(self, row: int, col: int) -> None:
        """Place the current player's stone at a 0-based board position."""
        if self.winner is not None or self.is_draw():
            raise ValueError("The game is already over.")
        if not self._in_bounds(row, col):
            raise ValueError("Move is out of bounds.")
        if self.board[row][col] != ".":
            raise ValueError("Cell is already occupied.")

        self.board[row][col] = self.current_player
        self.moves_played += 1

        if self.check_win(row, col):
            self.winner = self.current_player
            return

        if self.is_draw():
            return

        self.current_player = "O" if self.current_player == "X" else "X"

    def check_win(self, row: int, col: int) -> bool:
        """Check if the latest move at (row, col) creates a winning line."""
        player = self.board[row][col]
        if player == ".":
            return False

        directions: Tuple[Tuple[int, int], ...] = (
            (1, 0),
            (0, 1),
            (1, 1),
            (1, -1),
        )
        for dr, dc in directions:
            count = 1
            count += self._count_in_direction(row, col, dr, dc, player)
            count += self._count_in_direction(row, col, -dr, -dc, player)
            if count >= self.win_length:
                return True
        return False

    def is_draw(self) -> bool:
        """Return True when board is full and no winner exists."""
        return self.winner is None and self.moves_played == self.size * self.size

    def render(self) -> str:
        """Render the board as a printable string."""
        width = len(str(self.size))
        header = " " * (width + 1) + " ".join(f"{idx:>{width}}" for idx in range(1, self.size + 1))
        rows = []
        for idx, row in enumerate(self.board, start=1):
            rows.append(f"{idx:>{width}} " + " ".join(f"{cell:>{width}}" for cell in row))
        return "\n".join([header] + rows)

    def _count_in_direction(self, row: int, col: int, dr: int, dc: int, player: str) -> int:
        count = 0
        r, c = row + dr, col + dc
        while self._in_bounds(r, c) and self.board[r][c] == player:
            count += 1
            r += dr
            c += dc
        return count

    def _in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.size and 0 <= col < self.size


def parse_move(raw_move: str, size: int) -> Tuple[int, int]:
    """Parse move input from 1-based coordinates to 0-based indexes."""
    parts = raw_move.strip().split()
    if len(parts) != 2:
        raise ValueError("Please enter two numbers: row col")

    try:
        row = int(parts[0])
        col = int(parts[1])
    except ValueError as exc:
        raise ValueError("Row and column must be integers.") from exc

    row -= 1
    col -= 1
    if not (0 <= row < size and 0 <= col < size):
        raise ValueError(f"Coordinates must be between 1 and {size}.")
    return row, col


def run_cli() -> None:
    """Run an interactive two-player command line Gomoku game."""
    game = GomokuGame()
    print("Welcome to Gomoku!")
    print("Players take turns placing stones.")
    print("Input format: row col (1-based), or type q to quit.\n")

    while True:
        print(game.render())
        if game.winner is not None:
            print(f"\nPlayer {game.winner} wins!")
            break
        if game.is_draw():
            print("\nIt's a draw!")
            break

        move = input(f"\nPlayer {game.current_player}, enter your move: ").strip()
        if move.lower() in {"q", "quit", "exit"}:
            print("Game exited.")
            break

        try:
            row, col = parse_move(move, game.size)
            game.place_stone(row, col)
        except ValueError as exc:
            print(f"Invalid move: {exc}\n")


if __name__ == "__main__":
    run_cli()
