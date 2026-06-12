# Reward Normalization and Distributional Critic Value Range

## Reward Normalization

Implemented in `utils/normalization.py` via `RewardNormalizer`.

### Running Return Estimate

At each environment step (during training), a running discounted return `G` is maintained:

```
G_t = gamma * (1 - done) * G_{t-1} + r_t
```

The variance of `G` is tracked via a `RunningMeanStd` object (`G_rms`), and the running max of `|G|` is tracked as `G_r_max`.

### Scaling Denominator

Before passing rewards to the agent's update, each reward is divided by:

```
denominator = max(sqrt(Var(G) + eps), G_r_max / g_max)
```

- **`sqrt(Var(G) + eps)`**: standard deviation of the running return — pushes toward unit-variance returns.
- **`G_r_max / g_max`**: ensures the largest observed return never exceeds `g_max`. Acts as a hard cap.

The larger of the two is used, so the scaling is variance-based under normal conditions but clips extreme returns when they occur.

### Scaled Reward

```
r_scaled = r / denominator
```

Only rewards are scaled; the critic receives scaled rewards and computes targets in the normalized space.

---

## Distributional Critic Value Range

Configured in [configs/agent/simbaV2.yaml](../configs/agent/simbaV2.yaml):

```yaml
normalized_g_max: 5.0          # g_max passed to RewardNormalizer
critic_num_bins: 101
critic_min_v: -5.0             # = -normalized_g_max
critic_max_v:  5.0             # = +normalized_g_max
```

The critic uses a categorical distribution over 101 atoms linearly spaced on `[-5, 5]` (step size 0.1):

```
bin_values = linspace(-5.0, 5.0, 101)
```

### Why `[-5, 5]`?

The reward normalization guarantees that discounted returns stay within approximately `[-g_max, +g_max]`. Setting `critic_min_v = -g_max` and `critic_max_v = +g_max` ensures the critic's support covers the full range of normalized returns.

There is no per-environment override; HumanoidBench uses the same `[-5, 5]` range as all other domains.
