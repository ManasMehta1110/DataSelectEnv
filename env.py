import math
import numpy as np
import random

from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

from models import Observation, Action, EnvState
from sampling import sample_uncertainty, sample_diversity, sample_random, entropy
from reward import mean_cosine, running_mean


class DatasetState:
    def __init__(self, X_pool, y_pool, X_val, y_val, X_train, y_train, budget, y_all, noise_mask):
        self.X_pool = X_pool
        self.y_pool = y_pool
        self.X_val = X_val
        self.y_val = y_val
        self.X_train = X_train
        self.y_train = y_train
        self.budget = budget
        self.steps = 0
        self.performance = 0.0
        self.train_centroid = np.mean(X_train, axis=0)
        self.y_all = y_all
        self.noise_mask = noise_mask


class DataSelectEnv:
    WARMUP = 2

    def __init__(self, cfg, seed: int = 42):
        """
        cfg  — environment config dict (budget, max_steps, alpha, data)
        seed — random seed; stored here, applied at reset() so every
               episode is reproducible regardless of how many times
               the env has been constructed or stepped.
        """
        self.cfg = cfg
        self.seed = seed
        self._episode_state = None  # set by reset()

    # ------------------------------------------------------------------
    # Core OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self):
        """Start a new episode. Returns initial Observation."""
        # Seed here — ensures every reset() call produces identical data
        np.random.seed(self.seed)
        random.seed(self.seed)

        X, y = make_classification(
            n_samples=1500,
            n_features=20,
            n_informative=5,
            n_redundant=5,
            n_clusters_per_class=2,
            class_sep=1.0,
            flip_y=0.1,
            random_state=42,   # dataset skeleton is fixed; noise injection varies by seed
        )

        # Scaling — must happen before splits
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        X_seed, y_seed = X[:100],    y[:100]
        X_val,  y_val  = X[200:400], y[200:400]
        X_pool, y_pool = X[400:],    y[400:]
        X_pool = X_pool + np.random.normal(0, 0.5, X_pool.shape)

        # Noise injection — flip_y from task cfg overrides base value
        flip_prob = self.cfg["data"].get("flip_y", 0.1)
        noise_mask = np.random.rand(len(y_pool)) < flip_prob
        y_pool_noisy = y_pool.copy()
        # Guaranteed label flip (1-y), not random assignment.
        # Random assignment gives the correct label 50% of the time, halving
        # effective noise. Flipping guarantees every noise_mask sample is wrong.
        y_pool_noisy[noise_mask] = 1 - y_pool[noise_mask]
        X_pool[noise_mask] += np.random.normal(0, 0.1, X_pool[noise_mask].shape)

        # Fresh model every episode
        self.model = SGDClassifier(
            loss="log_loss",
            learning_rate="adaptive",
            eta0=0.01,
            max_iter=1,
            tol=None,
            random_state=42,
        )

        # Warm start on seed data
        for _ in range(10):
            self.model.partial_fit(X_seed, y_seed, classes=np.unique(y))

        self._episode_state = DatasetState(
            X_pool, y_pool_noisy, X_val, y_val,
            X_seed.copy(), y_seed.copy(),
            self.cfg["budget"], y, noise_mask,
        )

        # Noisy centroid — used in step() for the noise trap
        if np.any(noise_mask):
            self._episode_state.noisy_centroid = np.mean(X_pool[noise_mask], axis=0)
        else:
            self._episode_state.noisy_centroid = np.zeros(X_pool.shape[1])

        self._episode_state.performance = self._score()
        return self._obs()

    def step(self, action: Action):
        """Execute one action. Returns (Observation, reward, done, info)."""
        s = self._episode_state

        if s is None:
            raise RuntimeError("Call reset() before step().")

        # Pool empty guard
        if len(s.X_pool) == 0:
            return self._obs(), 0.0, True, {"reason": "pool_empty"}

        # Sanity check — noise_mask must always stay in sync with pool
        assert len(s.noise_mask) == len(s.X_pool), (
            f"noise_mask length {len(s.noise_mask)} != X_pool length {len(s.X_pool)}"
        )

        # Normalize strategy weights
        w = action.strategy_weights
        total = sum(w.values()) + 1e-8
        w = {k: v / total for k, v in w.items()}

        min_b = self.cfg.get("min_batch", 1)
        b = min(action.batch_size, s.budget)
        if b < min_b and action.action_type != "stop":
            b = min_b   # enforce minimum; prevents single-sample gaming
        if b <= 0:
            return self._obs(), -0.01, False, {"error": "empty batch"}

        if action.action_type == "stop":
            perf_threshold = self.cfg.get("stop_threshold", 0.60)
            if s.performance >= perf_threshold:
                stop_reward = 0.05 * s.budget
            else:
                stop_reward = -1.0
            return self._obs(), stop_reward, True, {}

        # Uncertainty + noise trap
        proba_pool = self.model.predict_proba(s.X_pool)
        H = entropy(proba_pool)

        # Noise trap: boost entropy of noisy samples so uncertainty sampling is
        # attracted to them. Capped at 0.55 (< log(2) ≈ 0.693 max binary entropy)
        # so clean uncertain samples can still compete — trap misleads rather
        # than completely overrides, keeping uncertainty a near-miss on hard.
        max_entropy = math.log(2)   # ≈ 0.693 for binary classifier
        flip_prob   = self.cfg["data"].get("flip_y", 0.1)
        boost_raw   = 0.1 + flip_prob * 2.0
        noise_boost = s.noise_mask.astype(float) * min(boost_raw, 0.55)
        H_adj = H + noise_boost

        # Sampling
        n_u = int(b * w.get("uncertainty", 0))
        n_d = int(b * w.get("diversity", 0))
        n_r = max(0, b - n_u - n_d)

        sel = set()
        sel.update(sample_uncertainty(s, H_adj, n_u, sel))
        sel.update(sample_diversity(s, n_d, sel))
        sel.update(sample_random(s, n_r, sel))

        idx = np.array(list(sel), dtype=int)
        if len(idx) == 0:
            return self._obs(), -0.01, False, {"error": "no samples selected"}

        Xb, yb = s.X_pool[idx], s.y_pool[idx]
        selected_noise = s.noise_mask[idx]
        noise_ratio = float(np.mean(selected_noise))

        # Remove selected samples from pool — keep noise_mask in sync
        keep = np.ones(len(s.X_pool), dtype=bool)
        keep[idx] = False
        s.X_pool     = s.X_pool[keep]
        s.y_pool     = s.y_pool[keep]
        s.noise_mask = s.noise_mask[keep]

        # Incremental training
        for _ in range(3):
            self.model.partial_fit(Xb, yb, classes=np.unique(s.y_all))

        old = s.performance
        new = self._score()

        # ----------------------------------------------------------------
        # Reward design
        # ----------------------------------------------------------------
        gain = (new - old) * 5.0

        # Distance-based diversity bonus: rewards batches that cover regions
        # far from existing training data. Diversity sampling scores high
        # (~0.25), random scores average (~0.22), uncertainty scores low
        # (~0.15) because boundary samples cluster near the centroid.
        diversity_bonus = float(np.mean(
            np.linalg.norm(Xb - s.train_centroid, axis=1)
        )) * 0.05
        gain += diversity_bonus

        redundancy = mean_cosine(Xb, s.train_centroid)
        if redundancy > 0.8:
            gain *= 0.5
        if new > 0.85:
            gain *= 0.7

        # Noise penalty scales with task difficulty: easy is forgiving,
        # hard severely punishes noisy selections.
        flip_prob = self.cfg["data"].get("flip_y", 0.1)
        noise_scale = 1.0 + flip_prob * 2.0   # 1.1 easy | 1.5 medium | 1.6 hard
        noise_penalty = noise_scale * noise_ratio
        reward = gain - 0.01 * b - 0.3 * redundancy - noise_penalty
        reward += 0.15   # baseline: keeps reward in mixed-sign territory so
                         # RL agents receive positive signal for good steps
        # ----------------------------------------------------------------

        # Update state
        s.train_centroid = running_mean(s.train_centroid, Xb)
        s.X_train  = np.vstack([s.X_train, Xb])
        s.y_train  = np.concatenate([s.y_train, yb])
        s.performance = new
        s.budget  -= b
        s.steps   += 1

        info = {
            "gain":        float(gain),
            "redundancy":  float(redundancy),
            "noise_ratio": float(noise_ratio),
        }

        done = (s.budget <= 0) or (s.steps >= self.cfg["max_steps"])
        return self._obs(), float(reward), done, info

    def get_state(self) -> EnvState:
        """
        OpenEnv spec state() requirement.
        Returns episode metadata — called by GET /state on the server.
        Named get_state() to avoid collision with the self._episode_state attribute.
        """
        if self._episode_state is None:
            return EnvState(
                step_count=0,
                remaining_budget=None,
                current_performance=None,
                pool_size=None,
                done=False,
            )
        s = self._episode_state
        return EnvState(
            step_count=int(s.steps),
            remaining_budget=int(s.budget),
            current_performance=float(s.performance),
            pool_size=int(len(s.X_pool)),
            done=(s.budget <= 0 or s.steps >= self.cfg["max_steps"]),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _score(self):
        p = self.model.predict_proba(self._episode_state.X_val)
        return max(0.05, float(1 / (1 + log_loss(self._episode_state.y_val, p))))

    def _obs(self):
        s = self._episode_state
        return Observation(
            remaining_budget=int(s.budget),
            diversity_score=float(np.std(s.X_train)),
            noise_estimate=float(np.mean(s.noise_mask)),
            current_performance=float(s.performance),
            samples_available=int(len(s.X_pool)),
        )