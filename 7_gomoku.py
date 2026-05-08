"""A terminal Gomoku (Five-in-a-Row) mini game.

- Human plays as X.
- Computer plays as O with a lightweight tactical AI.
- Board size defaults to 15x15.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

EMPTY = "."
HUMAN = "X"
AI = "O"

DIRECTIONS: Sequence[Tuple[int, int]] = (
    (1, 0),
    (0, 1),
    (1, 1),
    (1, -1),
)


@dataclass
class Gomoku:
    size: int = 15

    def __post_init__(self) -> None:
        self.board: List[List[str]] = [[EMPTY for _ in range(self.size)] for _ in range(self.size)]
        self.turn_count = 0

    def inside(self, row: int, col: int) -> bool:
        return 0 <= row < self.size and 0 <= col < self.size

    def is_empty(self, row: int, col: int) -> bool:
        return self.inside(row, col) and self.board[row][col] == EMPTY

    def place(self, row: int, col: int, stone: str) -> bool:
        if not self.is_empty(row, col):
            return False
        self.board[row][col] = stone
        self.turn_count += 1
        return True

    def remove(self, row: int, col: int) -> None:
        if self.board[row][col] != EMPTY:
            self.board[row][col] = EMPTY
            self.turn_count -= 1

    def is_full(self) -> bool:
        return self.turn_count >= self.size * self.size

    def _count_one_direction(self, row: int, col: int, dr: int, dc: int, stone: str) -> int:
        count = 0
        r, c = row + dr, col + dc
        while self.inside(r, c) and self.board[r][c] == stone:
            count += 1
            r += dr
            c += dc
        return count

    def line_len(self, row: int, col: int, dr: int, dc: int, stone: str) -> int:
        return (
            1
            + self._count_one_direction(row, col, dr, dc, stone)
            + self._count_one_direction(row, col, -dr, -dc, stone)
        )

    def has_five(self, row: int, col: int, stone: str) -> bool:
        return any(self.line_len(row, col, dr, dc, stone) >= 5 for dr, dc in DIRECTIONS)

    def empty_cells(self) -> Iterable[Tuple[int, int]]:
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r][c] == EMPTY:
                    yield r, c

    def candidate_cells(self) -> List[Tuple[int, int]]:
        if self.turn_count == 0:
            mid = self.size // 2
            return [(mid, mid)]

        candidates = set()
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r][c] == EMPTY:
                    continue
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        nr, nc = r + dr, c + dc
                        if self.is_empty(nr, nc):
                            candidates.add((nr, nc))

        if not candidates:
            return list(self.empty_cells())
        return list(candidates)

    def _shape_score(self, length: int) -> int:
        if length >= 5:
            return 100_000
        if length == 4:
            return 8_000
        if length == 3:
            return 500
        if length == 2:
            return 80
        return 5

    def score_move(self, row: int, col: int, stone: str) -> int:
        score = 0
        for dr, dc in DIRECTIONS:
            line = self.line_len(row, col, dr, dc, stone)
            score += self._shape_score(line)
        return score

    def best_ai_move(self) -> Optional[Tuple[int, int]]:
        candidates = self.candidate_cells()
        if not candidates:
            return None

        random.shuffle(candidates)

        # 1) Win immediately if possible.
        for r, c in candidates:
            self.place(r, c, AI)
            win_now = self.has_five(r, c, AI)
            self.remove(r, c)
            if win_now:
                return r, c

        # 2) Block immediate human win.
        for r, c in candidates:
            self.place(r, c, HUMAN)
            human_win = self.has_five(r, c, HUMAN)
            self.remove(r, c)
            if human_win:
                return r, c

        # 3) Evaluate tactical score.
        best_score = -1
        best_move: Optional[Tuple[int, int]] = None
        for r, c in candidates:
            self.place(r, c, AI)
            ai_score = self.score_move(r, c, AI)
            self.remove(r, c)

            self.place(r, c, HUMAN)
            human_pressure = self.score_move(r, c, HUMAN)
            self.remove(r, c)

            total = ai_score + int(0.9 * human_pressure)
            if total > best_score:
                best_score = total
                best_move = (r, c)

        return best_move

    def display(self) -> None:
        header = "   " + " ".join(f"{i:2d}" for i in range(self.size))
        print(header)
        for r in range(self.size):
            row_str = " ".join(f" {v}" for v in self.board[r])
            print(f"{r:2d} {row_str}")


def parse_move(raw: str, size: int) -> Optional[Tuple[int, int]]:
    raw = raw.strip().replace(",", " ")
    parts = [p for p in raw.split(" ") if p]
    if len(parts) != 2:
        return None
    try:
        r, c = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    if not (0 <= r < size and 0 <= c < size):
        return None
    return r, c


def run_cli(size: int) -> None:
    game = Gomoku(size=size)
    print("Welcome to Gomoku! You are X, AI is O.")
    print("Enter move as: row col (example: 7 7). Enter 'q' to quit.")

    while True:
        game.display()
        user_input = input("Your move: ").strip()
        if user_input.lower() in {"q", "quit", "exit"}:
            print("Game ended.")
            return

        move = parse_move(user_input, size)
        if move is None:
            print("Invalid move format. Use: row col")
            continue

        ur, uc = move
        if not game.place(ur, uc, HUMAN):
            print("That cell is occupied. Try another move.")
            continue

        if game.has_five(ur, uc, HUMAN):
            game.display()
            print("You win! Great job.")
            return

        if game.is_full():
            game.display()
            print("Draw: board is full.")
            return

        ai_move = game.best_ai_move()
        if ai_move is None:
            game.display()
            print("Draw: no valid move left.")
            return

        ar, ac = ai_move
        game.place(ar, ac, AI)
        print(f"AI move: {ar} {ac}")

        if game.has_five(ar, ac, AI):
            game.display()
            print("AI wins. Try again!")
            return

        if game.is_full():
            game.display()
            print("Draw: board is full.")
            return


def self_test() -> None:
    # Horizontal win check.
    g1 = Gomoku(10)
    for i in range(5):
        assert g1.place(3, i, HUMAN)
    assert g1.has_five(3, 2, HUMAN)

    # Vertical win check.
    g2 = Gomoku(10)
    for i in range(5):
        assert g2.place(i, 4, AI)
    assert g2.has_five(2, 4, AI)

    # Diagonal win check.
    g3 = Gomoku(10)
    for i in range(5):
        assert g3.place(i, i, HUMAN)
    assert g3.has_five(2, 2, HUMAN)

    # AI should block direct threat.
    g4 = Gomoku(10)
    for c in range(4):
        assert g4.place(6, c, HUMAN)
    move = g4.best_ai_move()
    assert move in {(6, 4)}, f"unexpected block move: {move}"

    # AI should finish immediate win.
    g5 = Gomoku(10)
    for c in range(4):
        assert g5.place(2, c, AI)
    move = g5.best_ai_move()
    assert move in {(2, 4)}, f"unexpected winning move: {move}"

    print("Self-test passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Gomoku in terminal.")
    parser.add_argument("--size", type=int, default=15, help="Board size (default: 15)")
    parser.add_argument("--self-test", action="store_true", help="Run internal tests")
    args = parser.parse_args()

    if args.size < 5:
        raise ValueError("Board size must be at least 5")

    if args.self_test:
        self_test()
        return

    run_cli(args.size)


if __name__ == "__main__":
    main()
