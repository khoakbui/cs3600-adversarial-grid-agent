from collections.abc import Callable
from typing import Tuple
import random
import math

import numpy as np

from game import board, move, enums


class PlayerAgent:
    """
    You may add and modify functions, however, __init__, commentate and play are the entry points for
    your program and should not be changed.
    """

    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        """
        Initialization code. We set up persistent state here so we can keep information
        across turns, especially our belief over where the rat might be.
        """
        self.size = getattr(enums, "BOARD_SIZE", 8)
        self.n_cells = self.size * self.size
        self.transition_matrix = self._prepare_transition_matrix(transition_matrix)
        self.turn_idx = 0

        # Belief over rat location as a flat vector of length 64.
        self.belief = np.ones(self.n_cells, dtype=np.float64) / self.n_cells

        # Heuristic weights. These are intentionally simple and stable.
        self.search_threshold = 0.42
        self.score_weight = 14.0
        self.mobility_weight = 0.35
        self.rat_distance_weight = 1.4
        self.center_weight = 0.10
        self.carpet_bonus_scale = 2.4
        self.prime_bonus = 1.2

        # Track whether a new rat likely spawned last turn.
        self.last_seen_player_search = None
        self.last_seen_opponent_search = None

        # Optional: if you want a reproducible fallback tie-break.
        self.rng = random.Random(3600)

    def commentate(self):
        """
        Optional end-of-game commentary.
        """
        peak = float(np.max(self.belief)) if self.belief is not None else 0.0
        return f"Turns played: {self.turn_idx}, peak rat belief: {peak:.3f}"

    def play(
        self,
        board: board.Board,
        sensor_data: Tuple,
        time_left: Callable,
    ):
        """
        Main decision function. Updates belief, decides whether searching is worth it,
        otherwise evaluates all legal movement moves and picks the best one.
        """
        self.turn_idx += 1

        # Keep belief consistent with any search outcome that may have happened previously.
        self._handle_search_feedback(board)

        # Update rat belief from current turn's noisy observation.
        self._update_belief(board, sensor_data)

        # If one location is sufficiently likely, search it.
        best_loc, best_prob = self._best_guess_location(board)
        if best_loc is not None and best_prob >= self._dynamic_search_threshold(board, time_left):
            return move.Move.search(best_loc)

        # Otherwise pick the best movement action.
        moves = board.get_valid_moves(exclude_search=True)
        if not moves:
            # Defensive fallback; should rarely happen.
            all_moves = board.get_valid_moves(exclude_search=False)
            if all_moves:
                return self.rng.choice(all_moves)
            return move.Move.search(best_loc if best_loc is not None else (0, 0))

        best_move = None
        best_value = -float("inf")

        # If we are under time pressure, skip some extra tie-break work.
        remaining = self._safe_time_left(board, time_left)
        low_time = remaining is not None and remaining < 10.0

        for m in moves:
            next_board = board.forecast_move(m)
            if next_board is None:
                continue

            value = self._evaluate_board(next_board)

            # Small direct move-type bonuses to guide behavior.
            if m.move_type == enums.MoveType.CARPET:
                value += self.carpet_bonus_scale * getattr(m, "roll_length", 0)
            elif m.move_type == enums.MoveType.PRIME:
                value += self.prime_bonus

            # Prefer moving closer to the current best rat guess.
            if best_loc is not None:
                my_pos = next_board.player_worker.get_location()
                d = self._manhattan(my_pos, best_loc)
                value -= self.rat_distance_weight * d

            # Slight bonus for central positions early/mid game.
            if not low_time:
                my_pos = next_board.player_worker.get_location()
                value -= self.center_weight * self._dist_to_center(my_pos)

            # Tiny deterministic jitter to avoid pathological ties.
            value += 1e-6 * self.rng.random()

            if value > best_value:
                best_value = value
                best_move = m

        if best_move is not None:
            return best_move

        return self.rng.choice(moves)

    # ------------------------------------------------------------------
    # Belief tracking
    # ------------------------------------------------------------------

    def _prepare_transition_matrix(self, transition_matrix):
        """
        Converts the provided transition matrix into a normalized NumPy array when possible.
        If unavailable, returns None and we will use a local fallback transition model.
        """
        if transition_matrix is None:
            return None
        try:
            T = np.asarray(transition_matrix, dtype=np.float64)
            if T.shape != (self.n_cells, self.n_cells):
                return None
            row_sums = T.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0.0] = 1.0
            T = T / row_sums
            return T
        except Exception:
            return None

    def _handle_search_feedback(self, board_obj):
        """
        Adjust belief based on any search result recorded on the board.
        If a rat was found, a new rat spawns, so reset to a broad prior.
        If a search failed, eliminate that searched square.
        """
        try:
            player_search = getattr(board_obj, "player_search", (None, False))
            opponent_search = getattr(board_obj, "opponent_search", (None, False))
        except Exception:
            return

        # Reset on a successful search by either side.
        for info in (player_search, opponent_search):
            loc, found = info
            if loc is not None and found:
                self._reset_belief_uniform()
                self.last_seen_player_search = player_search
                self.last_seen_opponent_search = opponent_search
                return

        # Eliminate searched square on a failed search if present.
        for info in (player_search, opponent_search):
            loc, found = info
            if loc is not None and not found:
                idx = self._loc_to_index(loc)
                if 0 <= idx < self.n_cells:
                    self.belief[idx] = 0.0

        self._apply_board_mask(board_obj)
        self._normalize_belief()

        self.last_seen_player_search = player_search
        self.last_seen_opponent_search = opponent_search

    def _update_belief(self, board_obj, sensor_data):
        """
        One HMM-style belief update:
        1) prediction using transition matrix
        2) correction using observation likelihood
        """
        # Prediction
        if self.transition_matrix is not None:
            # belief_next[j] = sum_i belief[i] * T[i,j]
            self.belief = self.belief @ self.transition_matrix
        else:
            self.belief = self._fallback_predict(self.belief)

        # Rat can be under blocked squares too, so we only eliminate worker-occupied squares.
        self._apply_board_mask(board_obj)

        # Observation update
        if sensor_data is not None and len(sensor_data) >= 2:
            noise_obs, dist_obs = sensor_data[0], sensor_data[1]
            obs_likelihood = np.ones(self.n_cells, dtype=np.float64)

            my_pos = board_obj.player_worker.get_location()

            for idx in range(self.n_cells):
                loc = self._index_to_loc(idx)

                # Cannot currently occupy a worker's square.
                if self._is_worker_square(board_obj, loc):
                    obs_likelihood[idx] = 0.0
                    continue

                cell_type = board_obj.get_cell(loc)
                p_noise = self._noise_likelihood(cell_type, noise_obs)

                true_dist = self._manhattan(my_pos, loc)
                p_dist = self._distance_likelihood(true_dist, dist_obs)

                obs_likelihood[idx] = p_noise * p_dist

            self.belief *= obs_likelihood

        self._apply_board_mask(board_obj)
        self._normalize_belief()

    def _reset_belief_uniform(self):
        self.belief = np.ones(self.n_cells, dtype=np.float64) / self.n_cells

    def _normalize_belief(self):
        total = float(np.sum(self.belief))
        if total <= 0.0 or not np.isfinite(total):
            self._reset_belief_uniform()
        else:
            self.belief /= total

    def _apply_board_mask(self, board_obj):
        """
        Rat cannot be on either worker square at the instant we reason about searching.
        """
        try:
            p1 = board_obj.player_worker.get_location()
            p2 = board_obj.opponent_worker.get_location()
            for loc in (p1, p2):
                idx = self._loc_to_index(loc)
                if 0 <= idx < self.n_cells:
                    self.belief[idx] = 0.0
        except Exception:
            pass

    def _best_guess_location(self, board_obj):
        """
        Returns the location with highest current rat probability and its probability.
        """
        masked = self.belief.copy()
        try:
            for loc in (
                board_obj.player_worker.get_location(),
                board_obj.opponent_worker.get_location(),
            ):
                idx = self._loc_to_index(loc)
                if 0 <= idx < self.n_cells:
                    masked[idx] = 0.0
        except Exception:
            pass

        idx = int(np.argmax(masked))
        prob = float(masked[idx])
        return self._index_to_loc(idx), prob

    def _dynamic_search_threshold(self, board_obj, time_left_fn):
        """
        Conservative early, a bit more aggressive later.
        Search has EV > 0 when p > 1/3, but we use a stricter threshold
        because moving can also score points.
        """
        turns_left = getattr(board_obj.player_worker, "turns_left", 40)
        threshold = self.search_threshold

        # As the game winds down, be slightly more willing to search.
        if turns_left <= 10:
            threshold -= 0.04
        if turns_left <= 5:
            threshold -= 0.03

        remaining = self._safe_time_left(board_obj, time_left_fn)
        if remaining is not None and remaining < 20.0:
            threshold -= 0.02

        return max(0.34, threshold)

    # ------------------------------------------------------------------
    # Move evaluation
    # ------------------------------------------------------------------

    def _evaluate_board(self, board_obj):
        """
        Static evaluation after our forecasted move.
        """
        my_score = board_obj.player_worker.get_points()
        opp_score = board_obj.opponent_worker.get_points()

        value = self.score_weight * (my_score - opp_score)

        # Mobility matters: more legal options next turn is generally better.
        try:
            my_mobility = len(board_obj.get_valid_moves(exclude_search=True))
            value += self.mobility_weight * my_mobility
        except Exception:
            pass

        # Reward being near existing primed lines and useful squares.
        my_pos = board_obj.player_worker.get_location()
        value += 0.8 * self._local_prime_potential(board_obj, my_pos)

        # Mild penalty for hugging corners too much unless necessary.
        value -= 0.12 * self._dist_to_center(my_pos)

        return value

    def _local_prime_potential(self, board_obj, loc):
        """
        Counts nearby primed squares in four directions. This helps value positions
        that can soon carpet for larger points.
        """
        score = 0
        x, y = loc
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        for dx, dy in directions:
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
            score += run * run  # favor longer contiguous runs
        return score

    # ------------------------------------------------------------------
    # Observation model
    # ------------------------------------------------------------------

    def _noise_likelihood(self, cell_type, noise_obs):
        """
        Noise table from the rules:
            Blocked: squeak 0.5, scratch 0.3, squeal 0.2
            Space:   squeak 0.7, scratch 0.15, squeal 0.15
            Primed:  squeak 0.1, scratch 0.8, squeal 0.1
            Carpet:  squeak 0.1, scratch 0.1, squeal 0.8
        """
        noise_key = self._normalize_noise(noise_obs)

        if cell_type == enums.Cell.BLOCKED:
            probs = {"squeak": 0.5, "scratch": 0.3, "squeal": 0.2}
        elif cell_type == enums.Cell.SPACE:
            probs = {"squeak": 0.7, "scratch": 0.15, "squeal": 0.15}
        elif cell_type == enums.Cell.PRIMED:
            probs = {"squeak": 0.1, "scratch": 0.8, "squeal": 0.1}
        elif cell_type == enums.Cell.CARPET:
            probs = {"squeak": 0.1, "scratch": 0.1, "squeal": 0.8}
        else:
            probs = {"squeak": 1 / 3, "scratch": 1 / 3, "squeal": 1 / 3}

        return probs.get(noise_key, 1e-6)

    def _distance_likelihood(self, true_dist, observed_dist):
        """
        Distance noise model from the rules:
            one less: 0.12
            correct:  0.70
            one more: 0.12
            two more: 0.06
        Observed distance is never below zero.
        """
        try:
            obs = int(observed_dist)
        except Exception:
            return 1.0

        likelihood = 0.0

        # correct
        if obs == true_dist:
            likelihood += 0.70

        # one less than actual
        if obs == max(0, true_dist - 1):
            likelihood += 0.12

        # one more than actual
        if obs == true_dist + 1:
            likelihood += 0.12

        # two more than actual
        if obs == true_dist + 2:
            likelihood += 0.06

        return max(likelihood, 1e-8)

    def _normalize_noise(self, noise_obs):
        """
        Tries to robustly map whatever noise object is provided into
        'squeak', 'scratch', or 'squeal'.
        """
        if noise_obs is None:
            return "squeak"

        # Enum-like object with .name
        name = getattr(noise_obs, "name", None)
        if isinstance(name, str):
            return name.lower()

        s = str(noise_obs).lower()
        if "squeak" in s:
            return "squeak"
        if "scratch" in s:
            return "scratch"
        if "squeal" in s:
            return "squeal"

        # Fallback: treat unknown as near-uniform by selecting one key.
        return "squeak"

    # ------------------------------------------------------------------
    # Transition fallback
    # ------------------------------------------------------------------

    def _fallback_predict(self, belief_vec):
        """
        Fallback transition if no matrix is available:
        uniform spread over staying put + cardinal neighbors inside the board.
        """
        out = np.zeros_like(belief_vec)
        for idx in range(self.n_cells):
            loc = self._index_to_loc(idx)
            nbrs = [loc]
            x, y = loc
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    nbrs.append((nx, ny))
            share = belief_vec[idx] / len(nbrs)
            for nloc in nbrs:
                out[self._loc_to_index(nloc)] += share
        return out

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _loc_to_index(self, loc):
        x, y = loc
        return y * self.size + x

    def _index_to_loc(self, idx):
        return (idx % self.size, idx // self.size)

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _dist_to_center(self, loc):
        cx = (self.size - 1) / 2.0
        cy = (self.size - 1) / 2.0
        return abs(loc[0] - cx) + abs(loc[1] - cy)

    def _is_worker_square(self, board_obj, loc):
        return (
            loc == board_obj.player_worker.get_location()
            or loc == board_obj.opponent_worker.get_location()
        )

    def _safe_time_left(self, board_obj, time_left_fn):
        """
        Safely gets remaining time if possible.
        """
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