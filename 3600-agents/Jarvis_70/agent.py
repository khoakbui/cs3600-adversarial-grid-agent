from collections.abc import Callable
from typing import Tuple
import random

from game import board, move, enums


class PlayerAgent:
    """
    1-ply opponent-aware restart bot.
    Strategy:
    - No search for now
    - Prefer direct scoring
    - Use shallow minimax:
        choose move that leads to the best board after opponent's best reply
    """

    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        self.size = getattr(enums, "BOARD_SIZE", 8)
        self.rng = random.Random(3600)
        self.turn_idx = 0

    def commentate(self):
        return f"turns={self.turn_idx}"

    def play(
        self,
        board: board.Board,
        sensor_data: Tuple,
        time_left: Callable,
    ):
        self.turn_idx += 1

        moves = board.get_valid_moves(exclude_search=True)
        if not moves:
            all_moves = board.get_valid_moves(exclude_search=False)
            return self.rng.choice(all_moves) if all_moves else move.Move.search((0, 0))

        best_move = None
        best_value = float("-inf")

        for my_move in moves:
            after_my_move = board.forecast_move(my_move)
            if after_my_move is None:
                continue

            value = self._minimax_one_ply(after_my_move)

            # Tiny deterministic jitter for stable tie-breaking
            value += 1e-6 * self.rng.random()

            if value > best_value:
                best_value = value
                best_move = my_move

        return best_move if best_move is not None else self.rng.choice(moves)

    # ------------------------------------------------------------------
    # 1-ply opponent-aware evaluation
    # ------------------------------------------------------------------

    def _minimax_one_ply(self, after_my_move: board.Board) -> float:
        """
        Evaluate the board after our move, assuming opponent gets one best reply.
        """
        # First evaluate immediate board in case opponent has no legal moves.
        immediate_value = self._evaluate_for_me(after_my_move)

        # Flip perspective so board.player_worker becomes the opponent.
        opp_board = after_my_move.get_copy()
        opp_board.reverse_perspective()

        opp_moves = opp_board.get_valid_moves(exclude_search=True)
        if not opp_moves:
            return immediate_value

        opponent_best = float("-inf")

        for opp_move in opp_moves:
            after_opp_move = opp_board.forecast_move(opp_move)
            if after_opp_move is None:
                continue

            # Reverse back so evaluation is from our perspective again.
            after_opp_move.reverse_perspective()
            val_for_opp_reply = self._evaluate_for_me(after_opp_move)

            if val_for_opp_reply > opponent_best:
                opponent_best = val_for_opp_reply

        # Since opponent chooses the reply that's worst for us,
        # we minimize their best resulting value.
        if opponent_best == float("-inf"):
            return immediate_value

        return opponent_best

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate_for_me(self, board_obj: board.Board) -> float:
        """
        Evaluate current board from our perspective.
        board_obj.player_worker is us.
        """
        my_score = board_obj.player_worker.get_points()
        opp_score = board_obj.opponent_worker.get_points()
        my_pos = board_obj.player_worker.get_location()

        value = 0.0

        # 1. Score difference matters most.
        value += 20.0 * (my_score - opp_score)

        # 2. Reward current mobility.
        try:
            legal_count = len(board_obj.get_valid_moves(exclude_search=True))
            value += 0.6 * legal_count
        except Exception:
            pass

        # 3. Reward immediate carpet opportunities next turn.
        best_carpet_len = self._best_available_carpet_length(board_obj)
        value += 8.0 * best_carpet_len
        value += 3.0 * self._carpet_points(best_carpet_len)

        # 4. Reward being in a good position to create future scoring.
        value += 2.0 * self._frontier_value(board_obj, my_pos)
        value += 1.8 * self._prime_line_potential(board_obj, my_pos)

        # 5. Prefer ending on SPACE so next turn can still PRIME.
        try:
            cell = board_obj.get_cell(my_pos)
            if cell == enums.Cell.SPACE:
                value += 4.0
            elif cell == enums.Cell.PRIMED:
                value -= 3.0
            elif cell == enums.Cell.CARPET:
                value -= 2.0
        except Exception:
            pass

        # 6. Mild early-game center preference.
        turns_left = getattr(board_obj.player_worker, "turns_left", 40)
        if turns_left > 10:
            value -= 0.15 * self._dist_to_center(my_pos)

        # 7. Penalize if opponent has strong immediate carpet reply potential.
        opp_reply_len = self._best_available_carpet_length_for_enemy(board_obj)
        value -= 7.0 * opp_reply_len
        value -= 2.5 * self._carpet_points(opp_reply_len)

        return value

    # ------------------------------------------------------------------
    # Feature helpers
    # ------------------------------------------------------------------

    def _best_available_carpet_length(self, board_obj: board.Board) -> int:
        """
        Best carpet move available for current player.
        """
        best = 0
        try:
            for m in board_obj.get_valid_moves(exclude_search=True):
                if m.move_type == enums.MoveType.CARPET:
                    best = max(best, getattr(m, "roll_length", 0))
        except Exception:
            pass
        return best

    def _best_available_carpet_length_for_enemy(self, board_obj: board.Board) -> int:
        """
        Best carpet move available for opponent from the same board state.
        """
        enemy_board = board_obj.get_copy()
        enemy_board.reverse_perspective()
        best = 0
        try:
            for m in enemy_board.get_valid_moves(exclude_search=True):
                if m.move_type == enums.MoveType.CARPET:
                    best = max(best, getattr(m, "roll_length", 0))
        except Exception:
            pass
        return best

    def _prime_line_potential(self, board_obj: board.Board, loc) -> float:
        """
        Reward being adjacent to contiguous primed runs.
        """
        x, y = loc
        total = 0.0

        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            run = 0
            cx, cy = x + dx, y + dy
            while 0 <= cx < self.size and 0 <= cy < self.size:
                try:
                    cell = board_obj.get_cell((cx, cy))
                except Exception:
                    break
                if cell == enums.Cell.PRIMED:
                    run += 1
                    cx += dx
                    cy += dy
                else:
                    break
            total += run * run

        return total

    def _frontier_value(self, board_obj: board.Board, loc) -> float:
        """
        Count nearby SPACE cells. This approximates future prime options.
        """
        x, y = loc
        score = 0.0

        for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                try:
                    cell = board_obj.get_cell((nx, ny))
                except Exception:
                    continue

                if cell == enums.Cell.SPACE:
                    score += 1.0
                elif cell == enums.Cell.PRIMED:
                    score += 0.35

        return score

    def _carpet_points(self, roll_length: int) -> int:
        table = getattr(enums, "CARPET_POINTS_TABLE", None)
        if table is not None:
            try:
                return table[roll_length]
            except Exception:
                pass

        defaults = {1: -1, 2: 2, 3: 4, 4: 6, 5: 10, 6: 15, 7: 21}
        return defaults.get(roll_length, 0)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _dist_to_center(self, loc) -> float:
        c = (self.size - 1) / 2.0
        return abs(loc[0] - c) + abs(loc[1] - c)