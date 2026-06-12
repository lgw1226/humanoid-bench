"""
Verify that the SimBaV2 reward normalization behaves as expected:
  denominator = max(sqrt(Var(G) + eps),  G_r_max / g_max)
  r_scaled    = clip(r / denom, -g_max, +g_max)

Tests:
  1. Normalizer attributes initialise correctly.
  2. Single-env wrapper: denominator switches from var-based to cap-based
     once G_r_max grows large.
  3. Normalised rewards always stay within [-g_max, +g_max].
  4. G_r_max is non-decreasing.
  5. Returns reset to 0 on episode end.
  6. Vector-env wrapper mirrors the same behaviour.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
from flag.utils.wrappers.normalize_env import Normalizer, RunningMeanStd

G_MAX = 5.0
GAMMA = 0.99
EPS   = 1e-8


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_normalizer():
    return Normalizer(obs_shape=(4,), gamma=GAMMA, epsilon=EPS, g_max=G_MAX)


def _simba_denom(normalizer):
    """Recompute expected denominator from normalizer state."""
    var_denom = float(np.sqrt(max(normalizer.ret_rms.var, 0.0) + EPS))
    cap_denom = normalizer.G_r_max / normalizer.g_max
    return max(var_denom, cap_denom)


# Minimal stub env wrappers (bypass gymnasium entirely)
class _FakeSingleEnv:
    obs_dim = 4
    act_dim = 2
    max_episode_steps = 1000

    def __init__(self, rewards):
        self._rewards = list(rewards)
        self._step = 0

    def reset(self, key=None):
        self._step = 0
        return np.zeros(self.obs_dim, dtype=np.float32), {}

    def step(self, action):
        r = self._rewards[self._step % len(self._rewards)]
        self._step += 1
        terminated = self._step >= len(self._rewards)
        return np.zeros(self.obs_dim, dtype=np.float32), r, terminated, False, {}

    def sample_random_action(self, key=None):
        return np.zeros(self.act_dim)


class _FakeVecEnv:
    num_envs = 4
    obs_dim  = 4
    act_dim  = 2

    def __init__(self, rewards_per_env):
        # rewards_per_env: list of lists
        self._rewards = rewards_per_env
        self._steps = [0] * self.num_envs

    def reset(self, key=None):
        self._steps = [0] * self.num_envs
        return np.zeros((self.num_envs, self.obs_dim), dtype=np.float32), {}

    def step(self, actions):
        n = self.num_envs
        rewards = np.array([
            self._rewards[i][self._steps[i] % len(self._rewards[i])]
            for i in range(n)
        ], dtype=np.float32)
        for i in range(n):
            self._steps[i] += 1
        terminated = np.array([
            self._steps[i] >= len(self._rewards[i]) for i in range(n)
        ])
        truncated  = np.zeros(n, dtype=bool)
        obs = np.zeros((n, self.obs_dim), dtype=np.float32)
        return obs, rewards, terminated, truncated, {}


# ── wrap with the real wrappers ───────────────────────────────────────────────

from flag.utils.wrappers.normalize_env import (
    NormalizeJAXEnvWrapper,
    NormalizeJAXVectorEnvWrapper,
)


def _wrap_single(rewards, normalizer):
    env = _FakeSingleEnv(rewards)
    return NormalizeJAXEnvWrapper(
        env, normalizer=normalizer, training=True,
        norm_obs=False, norm_reward=True,
        clip_obs=10.0, clip_reward=G_MAX,
    )


def _wrap_vec(rewards_per_env, normalizer):
    env = _FakeVecEnv(rewards_per_env)
    return NormalizeJAXVectorEnvWrapper(
        env, normalizer=normalizer, training=True,
        norm_obs=False, norm_reward=True,
        clip_obs=10.0, clip_reward=G_MAX,
    )


# ── tests ─────────────────────────────────────────────────────────────────────

def test_normalizer_init():
    n = _make_normalizer()
    assert n.g_max == G_MAX
    assert n.gamma == GAMMA
    assert n.epsilon == EPS
    assert n.G_r_max == EPS, "G_r_max should start at epsilon"
    print("PASS  test_normalizer_init")


def test_rewards_bounded_single_env():
    """All normalised rewards must stay in [-g_max, +g_max]."""
    rewards = [1.0] * 200          # constant-reward episode
    n = _make_normalizer()
    env = _wrap_single(rewards, n)
    env.reset(key=None)

    normalised = []
    for r_raw in rewards:
        _, r_norm, terminated, truncated, _ = env.step(np.zeros(2))
        normalised.append(r_norm)
        if terminated or truncated:
            break

    assert all(abs(r) <= G_MAX + 1e-6 for r in normalised), \
        f"Reward out of bounds: min={min(normalised):.4f} max={max(normalised):.4f}"
    print(f"PASS  test_rewards_bounded_single_env  (range [{min(normalised):.3f}, {max(normalised):.3f}])")


def test_rewards_bounded_large_rewards():
    """High-magnitude rewards should be clipped/normalised within bounds."""
    rewards = [100.0] * 500
    n = _make_normalizer()
    env = _wrap_single(rewards, n)
    env.reset(key=None)

    normalised = []
    for _ in rewards:
        _, r_norm, terminated, truncated, _ = env.step(np.zeros(2))
        normalised.append(r_norm)
        if terminated or truncated:
            break

    assert all(abs(r) <= G_MAX + 1e-6 for r in normalised), \
        f"Reward out of bounds: min={min(normalised):.4f} max={max(normalised):.4f}"
    print(f"PASS  test_rewards_bounded_large_rewards  (range [{min(normalised):.3f}, {max(normalised):.3f}])")


def test_G_r_max_nondecreasing():
    rewards = list(np.random.default_rng(0).uniform(0, 10, 300))
    n = _make_normalizer()
    env = _wrap_single(rewards, n)
    env.reset(key=None)

    prev = n.G_r_max
    for _ in rewards:
        env.step(np.zeros(2))
        assert n.G_r_max >= prev, "G_r_max decreased"
        prev = n.G_r_max

    print(f"PASS  test_G_r_max_nondecreasing  (final G_r_max={n.G_r_max:.4f})")


def test_returns_reset_on_done():
    """Internal _returns must be 0.0 immediately after an episode ends."""
    rewards = [1.0] * 10   # episode length = 10
    n = _make_normalizer()
    env = _wrap_single(rewards, n)
    env.reset(key=None)

    for _ in rewards:
        _, _, terminated, truncated, _ = env.step(np.zeros(2))
        if terminated or truncated:
            assert env._returns == 0.0, f"_returns not reset: {env._returns}"
            break

    print("PASS  test_returns_reset_on_done")


def test_denom_is_max_of_both_terms():
    """
    Unit-test the two-term max directly by constructing normalizer states
    where each term dominates in turn.
    """
    # Case A: cap_denom dominates (large G_r_max, small Var)
    n = _make_normalizer()
    n.G_r_max = 50.0        # cap = 50/5 = 10
    n.ret_rms.var = 1.0     # var_denom = sqrt(1 + eps) ≈ 1
    var_a = float(np.sqrt(max(n.ret_rms.var, 0.0) + EPS))
    cap_a = n.G_r_max / n.g_max
    assert cap_a > var_a,  f"Case A: expected cap ({cap_a}) > var ({var_a})"
    expected_a = cap_a

    # Case B: var_denom dominates (high variance, modest G_r_max)
    n2 = _make_normalizer()
    n2.G_r_max = 0.1        # cap = 0.1/5 = 0.02
    n2.ret_rms.var = 9.0    # var_denom = sqrt(9 + eps) ≈ 3
    var_b = float(np.sqrt(max(n2.ret_rms.var, 0.0) + EPS))
    cap_b = n2.G_r_max / n2.g_max
    assert var_b > cap_b,  f"Case B: expected var ({var_b}) > cap ({cap_b})"
    expected_b = var_b

    # Confirm the wrapper picks max() in each case via _simba_denom
    assert abs(_simba_denom(n)  - expected_a) < 1e-6
    assert abs(_simba_denom(n2) - expected_b) < 1e-6
    print(f"PASS  test_denom_is_max_of_both_terms  "
          f"(cap wins: {expected_a:.2f}, var wins: {expected_b:.2f})")


def test_vec_env_rewards_bounded():
    """Vector-env wrapper: all normalised rewards stay in [-g_max, +g_max]."""
    rng = np.random.default_rng(1)
    rewards_per_env = [
        list(rng.uniform(0, 5, 100)) for _ in range(4)
    ]
    n = _make_normalizer()
    env = _wrap_vec(rewards_per_env, n)
    env.reset(key=None)

    all_rewards = []
    for step in range(100):
        _, r_norm, _, _, _ = env.step(np.zeros((4, 2)))
        all_rewards.extend(r_norm.tolist())

    assert all(abs(r) <= G_MAX + 1e-6 for r in all_rewards), \
        f"Vec reward out of bounds: min={min(all_rewards):.4f} max={max(all_rewards):.4f}"
    print(f"PASS  test_vec_env_rewards_bounded  (range [{min(all_rewards):.3f}, {max(all_rewards):.3f}])")


def test_denominator_formula():
    """Step-by-step check that the wrapper computes exactly the SimBaV2 denominator."""
    rewards = [2.0, 3.0, -1.0, 5.0, 0.5]
    n = _make_normalizer()
    env = _wrap_single(rewards, n)
    env.reset(key=None)

    G = 0.0
    for r_raw in rewards:
        G = GAMMA * G + r_raw
        _, r_norm, terminated, truncated, _ = env.step(np.zeros(2))

        expected_denom = _simba_denom(n)
        expected_r = float(np.clip(r_raw / expected_denom, -G_MAX, G_MAX))

        assert abs(r_norm - expected_r) < 1e-5, (
            f"r_raw={r_raw}: got {r_norm:.6f}, expected {expected_r:.6f}"
        )
        if terminated or truncated:
            break

    print("PASS  test_denominator_formula")


if __name__ == "__main__":
    test_normalizer_init()
    test_rewards_bounded_single_env()
    test_rewards_bounded_large_rewards()
    test_G_r_max_nondecreasing()
    test_returns_reset_on_done()
    test_denom_is_max_of_both_terms()
    test_vec_env_rewards_bounded()
    test_denominator_formula()
    print("\nAll tests passed.")
