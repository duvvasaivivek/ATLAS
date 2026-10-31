"""Pure, offline Tier 2 policies. No hardware access or clock setters."""

from copy import deepcopy
from dataclasses import asdict, dataclass
import math

from .experiment import digest

SPEC_VERSION = "frequency-1.0"
POLICIES = ("paper_additive", "discounted_loss_v1")


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def fraction(value):
    return finite(value) and 0 <= value <= 1


def log_value(value):
    """Preserve rejected nonfinite values in strict JSON without treating them as zero."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": str(value)}
    if isinstance(value, dict):
        return {k: log_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [log_value(v) for v in value]
    return value


@dataclass(frozen=True)
class Config:
    pairs: tuple
    policy: str = "paper_additive"
    alpha_core: float = .15
    alpha_memory: float = .02
    gamma: float = .3
    beta: float = .2
    discount: float = .95
    provenance: str = "synthetic_assumed"

    def __post_init__(self):
        pairs = tuple(tuple(p) for p in self.pairs)
        if any(len(p) != 2 or any(type(f) is not int or f <= 0 for f in p) for p in pairs):
            raise ValueError("Pairs must contain positive integer core/memory MHz")
        if len(set(pairs)) != len(pairs):
            raise ValueError("Duplicate frequency pairs")
        object.__setattr__(self, "pairs", tuple(sorted(pairs)))
        if self.policy not in POLICIES:
            raise ValueError("Unknown frequency policy")
        if not all(fraction(v) for v in (self.alpha_core, self.alpha_memory, self.gamma)):
            raise ValueError("Alpha and gamma must be finite fractions")
        if not fraction(self.beta) or not 0 < self.beta < 1:
            raise ValueError("Beta must be strictly between zero and one")
        if not fraction(self.discount) or not 0 < self.discount < 1:
            raise ValueError("Discount must be strictly between zero and one")
        if self.provenance not in ("synthetic_assumed", "hardware_reported_unverified"):
            raise ValueError("Phase 7 does not certify hardware actuation")

    @property
    def identity(self):
        return digest(dict(spec_version=SPEC_VERSION, **asdict(self)))


def target(frequency, levels):
    """Linear-in-MHz ATLAS interpretation, not ordinal level spacing."""
    low, high = min(levels), max(levels)
    return 0. if low == high else (frequency-low)/(high-low)


def loss(utilization, utilization_target, alpha):
    if not all(fraction(x) for x in (utilization, utilization_target, alpha)):
        raise ValueError("Loss inputs must be finite normalized fractions")
    return alpha*max(0., utilization_target-utilization) + (1-alpha)*max(0., utilization-utilization_target)


def losses(config, core, memory):
    if not fraction(core) or not fraction(memory):
        raise ValueError("Both utilization observations are required")
    cores = {p[0] for p in config.pairs}
    memories = {p[1] for p in config.pairs}
    return [config.gamma*(loss(core, target(c, cores), config.alpha_core) if len(cores)>1 else 0.)
            + (1-config.gamma)*(loss(memory, target(m, memories), config.alpha_memory) if len(memories)>1 else 0.)
            for c, m in config.pairs]


def initial_state(config, run_id):
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("A nonempty run ID is required")
    return dict(run_id=run_id, config_hash=config.identity, weights=[0.]*len(config.pairs),
                selected=0 if config.pairs else None, updates=0, last_sample=-1,
                last_epoch=-1, last_window=None)


def choose(weights, current, pairs):
    best = max(weights)
    tied = [i for i, w in enumerate(weights) if w == best]
    return current if current in tied else max(tied, key=lambda i: pairs[i])


def observation_errors(observation, state):
    errors = []
    if observation.get("run_id") != state["run_id"]:
        errors.append("cross_run")
    for field in ("sample_id", "epoch"):
        if type(observation.get(field)) is not int or observation[field] < 0:
            errors.append("invalid_"+field)
    if observation.get("valid") is not True:
        errors.append("source_invalid")
    if not all(fraction(observation.get(k)) for k in ("core", "memory")):
        errors.append("missing_or_invalid_utilization")
    if not isinstance(observation.get("source_ref"), str) or not observation["source_ref"]:
        errors.append("missing_source_reference")
    semantics = observation.get("semantics")
    if semantics not in ("synthetic_busy_fraction", "nvml_kernel_and_memory_busy_time_fraction"):
        errors.append("unsupported_metric_semantics")
    window = observation.get("core_window")
    if not isinstance(window, dict) or window != observation.get("memory_window"):
        errors.append("incompatible_windows")
        return errors
    kind = window.get("kind")
    end = window.get("end")
    if type(end) is not int or end < 0:
        errors.append("invalid_window_end")
    if kind == "synthetic_interval":
        start = window.get("start")
        if type(start) is not int or type(end) is not int or not 0 <= start < end:
            errors.append("invalid_window_interval")
        if semantics != "synthetic_busy_fraction":
            errors.append("incompatible_window_semantics")
    elif kind == "nvml_buffer_timestamp_us":
        if semantics != "nvml_kernel_and_memory_busy_time_fraction" or window.get("start") is not None:
            errors.append("invalid_nvml_window")
    else:
        errors.append("unqualified_window_kind")
    previous = state["last_window"]
    if previous is not None and not errors:
        if kind != previous["kind"]:
            errors.append("changed_clock_domain")
        elif end <= previous["end"] or (kind == "synthetic_interval" and window["start"] < previous["end"]):
            errors.append("stale_or_overlapping_window")
    if type(observation.get("epoch")) is int and observation["epoch"] < state["last_epoch"]:
        errors.append("stale_epoch")
    return errors


def transition(config, state, observation):
    """One independent valid sample updates every explicit candidate; rejected input holds."""
    if state.get("config_hash") != config.identity:
        raise ValueError("Cannot mix configurations within a run")
    if (len(state["weights"]) != len(config.pairs) or
            not all(finite(w) for w in state["weights"]) or
            (config.pairs and (type(state["selected"]) is not int or not 0 <= state["selected"] < len(config.pairs)))):
        raise ValueError("Invalid controller state")
    post = deepcopy(state)
    errors = observation_errors(observation, state)
    candidate_losses = []
    if not config.pairs:
        reason = "unsupported_no_candidates"
    elif errors:
        reason = "hold_invalid_observation"
    elif observation["sample_id"] <= state["last_sample"]:
        reason = "hold_duplicate_or_stale_sample"
    else:
        candidate_losses = losses(config, observation["core"], observation["memory"])
        if config.policy == "paper_additive":
            weights = [w + (1-(1-config.beta)*v) for w, v in zip(state["weights"], candidate_losses)]
        else:
            # Explicit extension: negative exponentially discounted cumulative loss.
            weights = [config.discount*w-v for w, v in zip(state["weights"], candidate_losses)]
        if not all(finite(w) for w in weights):
            raise ArithmeticError("Score overflow; start a separately identified run")
        post.update(weights=weights, selected=choose(weights, state["selected"], config.pairs),
                    updates=state["updates"]+1, last_sample=observation["sample_id"],
                    last_epoch=observation["epoch"], last_window=deepcopy(observation["core_window"]))
        reason = "recommend_pair"
    selected = post["selected"]
    return dict(spec_version=SPEC_VERSION, policy=config.policy, config_hash=config.identity,
                run_id=state["run_id"], observation=log_value(deepcopy(observation)),
                input_valid=not errors, errors=errors, reason=reason, losses=candidate_losses,
                pre_state=deepcopy(state), post_state=post,
                recommended_pair=list(config.pairs[selected]) if selected is not None else None,
                requested_pair=None, acknowledged_pair=None,
                observed_pair=deepcopy(observation.get("observed_pair")), actuation="offline_none",
                reduced_actuation=bool(config.pairs) and any(len({p[d] for p in config.pairs})==1 for d in (0,1)))
