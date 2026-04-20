from collections.abc import Callable
from typing import Tuple
import random
import numpy as np

from game import board, move, enums

CARPET_POINTS = {1: -1, 2: 2, 3: 4, 4: 6, 5: 10, 6: 15, 7: 21}


class PlayerAgent:

    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        self.size = getattr(enums, "BOARD_SIZE", 8)
        self.n_cells = self.size * self.size
        self.rng = random.Random(3600)
        self.turn_idx = 0
        self.searched_locs = set()

        # HMM belief distribution over all 64 cells
        self.belief = np.ones(self.n_cells, dtype=np.float64) / self.n_cells
        self.transition_matrix = self._prepare_transition(transition_matrix)

        # Search threshold: EV of search = 6p - 2 > 0 when p > 0.33
        self.search_threshold = 0.35

    def commentate(self):
        peak = float(np.max(self.belief))
        return f"turns={self.turn_idx}, peak_belief={peak:.3f}"

    def play(self, board, sensor_data: Tuple, time_left: Callable):
        self.turn_idx += 1

        # Step 1: Update rat belief with search feedback and sensor observation
        self._handle_search_feedback(board)
        self._update_belief(board, sensor_data)

        # Step 2: Decide whether to search for the rat
        best_rat_loc, best_prob = self._best_rat_guess(board)
        turns_left = getattr(board.player_worker, "turns_left", 40)
        threshold = max(0.32, self.search_threshold - (0.05 if turns_left <= 10 else 0))
        print(f"t={self.turn_idx} prob={best_prob:.3f} thr={threshold:.3f} belief_max={self.belief.max():.3f}")
        if best_prob >= threshold:
            return move.Move.search(best_rat_loc)

        # Step 3: Pick the best movement action via minimax
        moves = board.get_valid_moves(exclude_search=True)
        if not moves:
            all_moves = board.get_valid_moves(exclude_search=False)
            return self.rng.choice(all_moves) if all_moves else move.Move.search((0, 0))

        # Use depth-2 minimax when time allows, otherwise depth-1
        remaining = self._safe_time(board, time_left)
        depth = 2 if (remaining is None or remaining > 60.0) else 1

        moves = self._order_moves(moves)

        best_move = None
        best_value = float("-inf")

        for my_move in moves:
            # Carpet roll of length 1 scores -1 point; always skip it
            if my_move.move_type == enums.MoveType.CARPET:
                if getattr(my_move, "roll_length", 0) < 2:
                    continue

            after_my_move = board.forecast_move(my_move)
            if after_my_move is None:
                continue

            if depth >= 2:
                value = self._minimax_one_ply(after_my_move, best_rat_loc)
            else:
                value = self._evaluate_for_me(after_my_move, best_rat_loc)

            # Small jitter to break ties deterministically
            value += 1e-6 * self.rng.random()

            if value > best_value:
                best_value = value
                best_move = my_move

        return best_move if best_move is not None else self.rng.choice(moves)

    # ------------------------------------------------------------------
    # Minimax
    # ------------------------------------------------------------------

    def _minimax_one_ply(self, after_my_move, best_rat_loc) -> float:
        """
        My move is already applied. Simulate the opponent's best reply (MIN node).
        Returns the board value from our perspective after both moves.
        """
        immediate_value = self._evaluate_for_me(after_my_move, best_rat_loc)

        opp_board = after_my_move.get_copy()
        opp_board.reverse_perspective()

        opp_moves = opp_board.get_valid_moves(exclude_search=True)
        if not opp_moves:
            return immediate_value

        worst_for_me = float("inf")
        for opp_m in opp_moves:
            # Skip losing carpet moves for opponent too
            if opp_m.move_type == enums.MoveType.CARPET:
                if getattr(opp_m, "roll_length", 0) < 2:
                    continue

            after_opp = opp_board.forecast_move(opp_m)
            if after_opp is None:
                continue

            # Flip back to our perspective before evaluating
            after_opp.reverse_perspective()
            val = self._evaluate_for_me(after_opp, best_rat_loc)
            if val < worst_for_me:
                worst_for_me = val

        return worst_for_me if worst_for_me != float("inf") else immediate_value

    def _order_moves(self, moves):
        """Sort moves: highest-value carpet first, then prime, then plain."""
        def priority(m):
            if m.move_type == enums.MoveType.CARPET:
                return -CARPET_POINTS.get(getattr(m, "roll_length", 0), 0)
            if m.move_type == enums.MoveType.PRIME:
                return -0.5
            return 0
        return sorted(moves, key=priority)

    # ------------------------------------------------------------------
    # Board evaluation
    # ------------------------------------------------------------------

    def _evaluate_for_me(self, board_obj, best_rat_loc=None) -> float:
        """Static evaluation of the board from our perspective."""
        my_score = board_obj.player_worker.get_points()
        opp_score = board_obj.opponent_worker.get_points()
        my_pos = board_obj.player_worker.get_location()
        turns_left = getattr(board_obj.player_worker, "turns_left", 40)

        value = 0.0

        # Score difference is the primary objective
        value += 12.0 * (my_score - opp_score)

        # Reward having a long carpet available right now
        best_carpet_len = self._best_carpet_length(board_obj)
        if best_carpet_len >= 2:
            value += 6.0 * CARPET_POINTS.get(best_carpet_len, 0)

        # Reward being adjacent to long primed runs (carpet-ready lines)
        value += 3.0 * self._prime_line_potential(board_obj, my_pos)

        # Reward proximity to future scoring opportunities
        value += 2.0 * self._cell_potential(board_obj, my_pos)

        # Penalize opponent having a strong carpet available
        opp_carpet_len = self._best_carpet_length_for_opp(board_obj)
        if opp_carpet_len >= 2:
            value -= 5.0 * CARPET_POINTS.get(opp_carpet_len, 0)

        # Reward having more legal moves (mobility)
        try:
            value += 0.4 * len(board_obj.get_valid_moves(exclude_search=True))
        except Exception:
            pass

        # Prefer landing on SPACE so we can prime next turn
        try:
            cell = board_obj.get_cell(my_pos)
            if cell == enums.Cell.SPACE:
                value += 3.0
            elif cell == enums.Cell.PRIMED:
                value -= 4.0  # Standing on primed blocks future priming
            elif cell == enums.Cell.CARPET:
                value -= 1.0
        except Exception:
            pass

        # Mild center preference in the early/mid game
        if turns_left > 15:
            value -= 0.2 * self._dist_to_center(my_pos)

        # Move toward the most likely rat location
        if best_rat_loc is not None:
            rat_dist = abs(my_pos[0] - best_rat_loc[0]) + abs(my_pos[1] - best_rat_loc[1])
            value += 1.2 / (rat_dist + 1)

        return value

    # ------------------------------------------------------------------
    # Feature helpers
    # ------------------------------------------------------------------

    def _best_carpet_length(self, board_obj) -> int:
        """Returns the longest carpet roll (>= 2) available to the current player."""
        best = 0
        try:
            for m in board_obj.get_valid_moves(exclude_search=True):
                if m.move_type == enums.MoveType.CARPET:
                    length = getattr(m, "roll_length", 0)
                    if length >= 2:
                        best = max(best, length)
        except Exception:
            pass
        return best

    def _best_carpet_length_for_opp(self, board_obj) -> int:
        """Returns the longest carpet roll (>= 2) available to the opponent."""
        best = 0
        try:
            opp_board = board_obj.get_copy()
            opp_board.reverse_perspective()
            for m in opp_board.get_valid_moves(exclude_search=True):
                if m.move_type == enums.MoveType.CARPET:
                    length = getattr(m, "roll_length", 0)
                    if length >= 2:
                        best = max(best, length)
        except Exception:
            pass
        return best

    def _prime_line_potential(self, board_obj, loc) -> float:
        """
        Scores the primed runs adjacent to loc.
        Longer runs score higher because they enable bigger carpet rolls.
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
            if run >= 2:
                total += CARPET_POINTS.get(run, 0)
            elif run == 1:
                total += 0.5
        return total

    def _cell_potential(self, board_obj, loc, radius: int = 3) -> float:
        """
        Weighted sum of SPACE and PRIMED cells within a given radius.
        Cells closer to loc receive higher weight (1 / manhattan distance).
        """
        x, y = loc
        total = 0.0
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < self.size and 0 <= ny < self.size):
                    continue
                dist = abs(dx) + abs(dy)
                if dist == 0:
                    continue
                try:
                    cell = board_obj.get_cell((nx, ny))
                except Exception:
                    continue
                if cell == enums.Cell.SPACE:
                    total += 1.5 / dist
                elif cell == enums.Cell.PRIMED:
                    total += 3.0 / dist
        return total

    # ------------------------------------------------------------------
    # HMM belief tracking
    # ------------------------------------------------------------------

    def _prepare_transition(self, T):
        """Normalize and validate the transition matrix provided by the engine."""
        if T is None:
            return None
        try:
            T = np.asarray(T, dtype=np.float64)
            if T.shape != (self.n_cells, self.n_cells):
                return None
            row_sums = T.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            return T / row_sums
        except Exception:
            return None

    def _update_belief(self, board_obj, sensor_data):
        """
        One step of HMM belief update:
        1. Prediction: propagate belief through the transition matrix.
        2. Observation: reweight by the likelihood of the sensor reading.
        """
        # Prediction step
        if self.transition_matrix is not None:
            self.belief = self.belief @ self.transition_matrix
        else:
            self.belief = self._fallback_predict()

        # The rat cannot be on either worker's square
        for loc in (board_obj.player_worker.get_location(),
                    board_obj.opponent_worker.get_location()):
            self.belief[loc[1] * self.size + loc[0]] = 0.0

        # Observation update using noise and distance sensor
        if sensor_data is not None and len(sensor_data) >= 2:
            noise_obs, dist_obs = sensor_data[0], sensor_data[1]
            my_pos = board_obj.player_worker.get_location()

            # Sound likelihood table from the assignment spec
            NOISE_TABLE = {
                enums.Cell.BLOCKED: {"squeak": 0.5, "scratch": 0.3, "squeal": 0.2},
                enums.Cell.SPACE:   {"squeak": 0.7, "scratch": 0.15, "squeal": 0.15},
                enums.Cell.PRIMED:  {"squeak": 0.1, "scratch": 0.8,  "squeal": 0.1},
                enums.Cell.CARPET:  {"squeak": 0.1, "scratch": 0.1,  "squeal": 0.8},
            }
            # Distance noise model from the assignment spec
            DIST_TABLE = {-1: 0.12, 0: 0.70, 1: 0.12, 2: 0.06}

            noise_key = self._parse_noise(noise_obs)
            try:
                obs_dist = int(dist_obs)
            except Exception:
                obs_dist = None

            for idx in range(self.n_cells):
                x, y = idx % self.size, idx // self.size
                cell = board_obj.get_cell((x, y))
                p_noise = NOISE_TABLE.get(cell, {}).get(noise_key, 1.0 / 3)
                if obs_dist is not None:
                    true_dist = abs(x - my_pos[0]) + abs(y - my_pos[1])
                    p_dist = DIST_TABLE.get(obs_dist - true_dist, 1e-4)
                else:
                    p_dist = 1.0
                self.belief[idx] *= max(p_noise * p_dist, 1e-9)

        total = self.belief.sum()
        if total <= 0 or not np.isfinite(total):
            self.belief = np.ones(self.n_cells) / self.n_cells
        else:
            self.belief /= total

    def _handle_search_feedback(self, board_obj):
        """
        Adjust belief based on recorded search outcomes.
        - If the rat was found (by either player), reset to uniform prior.
        - If a search failed, zero out that cell.
        """
        for info in (board_obj.player_search, board_obj.opponent_search):
            loc, found = info
            if loc is not None and found:
                self.belief = np.ones(self.n_cells) / self.n_cells
                self.searched_locs = set()  # 추가
                return

        changed = False
        for info in (board_obj.player_search, board_obj.opponent_search):
            loc, found = info
            if loc is not None and not found:
                idx = loc[1] * self.size + loc[0]
                if 0 <= idx < self.n_cells:
                    self.belief[idx] = 0.0
                    changed = True

        # Normalize after zeroing out failed cells
        if changed:
            total = self.belief.sum()
            if total > 0:
                self.belief /= total

    def _best_rat_guess(self, board_obj):
        """Returns (location, probability) of the most likely rat cell."""
        masked = self.belief.copy()
        for loc in (board_obj.player_worker.get_location(),
                    board_obj.opponent_worker.get_location()):
            masked[loc[1] * self.size + loc[0]] = 0.0
        idx = int(np.argmax(masked))
        return (idx % self.size, idx // self.size), float(masked[idx])

    def _parse_noise(self, noise_obs) -> str:
        """Robustly convert the noise observation to 'squeak', 'scratch', or 'squeal'."""
        if noise_obs is None:
            return "squeak"
        name = getattr(noise_obs, "name", None)
        if isinstance(name, str):
            return name.lower()
        s = str(noise_obs).lower()
        for k in ("squeak", "scratch", "squeal"):
            if k in s:
                return k
        return "squeak"

    def _fallback_predict(self):
        """
        Fallback transition when no matrix is available.
        Spreads belief uniformly to cardinal neighbors and staying in place.
        """
        out = np.zeros(self.n_cells)
        for idx in range(self.n_cells):
            x, y = idx % self.size, idx // self.size
            nbrs = [(x, y)]
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    nbrs.append((nx, ny))
            share = self.belief[idx] / len(nbrs)
            for nx, ny in nbrs:
                out[ny * self.size + nx] += share
        return out

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _dist_to_center(self, loc) -> float:
        c = (self.size - 1) / 2.0
        return abs(loc[0] - c) + abs(loc[1] - c)

    def _safe_time(self, board_obj, time_left_fn):
        """Safely retrieve remaining time without crashing."""
        try:
            if callable(time_left_fn):
                val = time_left_fn()
                if val is not None:
                    return float(val)
        except Exception:
            pass
        try:
            return float(board_obj.player_worker.time_left)
        except Exception:
            return None