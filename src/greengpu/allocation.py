"""Pure Tier 1 transitions. No clocks, I/O, hardware calls, or hidden state."""

from bisect import bisect_left
from copy import deepcopy
from dataclasses import asdict, dataclass
import math

from .experiment import digest

POLICIES = ("paper_simple", "paper_adaptive", "paper_binary_literal",
            "bracketed_bisection", "paper_proportional", "robust_proportional_v1")


@dataclass(frozen=True)
class Config:
    policy: str
    n: int
    legal_counts: tuple | None = None
    step: float = .05
    adaptive_step: float = .15
    precision: float = .01
    budget: int = 200
    deadband: float = .05
    alpha: float = .25
    damping: float = .5
    max_move: float = .10
    dwell: int = 3
    phase_factor: float = 2.0
    phase_patience: int = 2

    def __post_init__(self):
        if self.policy not in (*POLICIES, "static") or type(self.n) is not int or self.n < 1:
            raise ValueError("Invalid policy or work size")
        if self.legal_counts is not None:
            if (not self.legal_counts or tuple(sorted(set(self.legal_counts))) != self.legal_counts
                    or any(type(c) is not int or not 0 <= c <= self.n for c in self.legal_counts)):
                raise ValueError("Legal counts must be sorted unique integers in [0,N]")
        for value in (self.step, self.adaptive_step, self.precision, self.deadband,
                      self.alpha, self.damping, self.max_move):
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError("Invalid ratio parameter")
        if (any(type(v) is not int or v < 1 for v in (self.budget, self.dwell, self.phase_patience))
                or not math.isfinite(self.phase_factor) or self.phase_factor <= 1):
            raise ValueError("Invalid policy budget/phase parameters")

    def counts(self, interior=False):
        values = self.legal_counts if self.legal_counts is not None else range(self.n + 1)
        if not interior:
            return values
        if self.legal_counts is None:
            return range(1, self.n)
        return tuple(c for c in values if 0 < c < self.n)

    def nearest(self, ratio, interior=False):
        values = self.counts(interior)
        if not values:
            raise ValueError("No legal partition")
        target = max(0., min(1., ratio)) * self.n
        pos = bisect_left(values, target)
        choices = values[max(0, pos-1):min(len(values), pos+1)]
        return min(choices, key=lambda c: (abs(c-target), c))


def initial_state(run_id):
    return dict(run_id=run_id, epoch=None, last_iteration=-1, eligible=0,
                step=None, sign=0, frozen=None, previous_ratio=None, bracket=None,
                filter=None, dwell=0, phase_hits=0)


def transition(config, state, observation):
    """Return an auditable decision; never mutate arguments, including on rejection."""
    before = deepcopy(state)
    after = deepcopy(state)
    o = observation
    count = o.get("cpu_items")
    if type(count) is not int or count not in config.counts():
        raise ValueError("Observation has illegal realized work count")
    ratio = count / config.n

    def emit(reason, target=None, selected=None):
        target = ratio if target is None else target
        selected = config.nearest(target) if selected is None else selected
        return dict(specification_version="1.1", policy=config.policy,
                    config_hash=digest(asdict(config)), observation=deepcopy(o),
                    pre_state=before, post_state=after, reason=reason,
                    requested_ratio=target, cpu_items=selected,
                    realized_ratio=selected/config.n)

    identity_ok = (o.get("run_id") == state["run_id"] and type(o.get("iteration")) is int
                   and o["iteration"] > state["last_iteration"]
                   and type(o.get("epoch")) is int and o["epoch"] >= 0
                   and (state["epoch"] is None or o["epoch"] >= state["epoch"]))
    if not identity_ok or o.get("valid") is not True or o.get("common_origin") is not True:
        return emit("hold_invalid_observation")
    a, b = o.get("t_cpu"), o.get("t_gpu")
    positive = lambda v: type(v) in (int, float) and math.isfinite(v) and v > 0
    if ((count and not positive(a)) or (count < config.n and not positive(b))
            or (not count and a is not None) or (count == config.n and b is not None)):
        return emit("hold_invalid_observation")
    if config.policy == "static":
        after.update(epoch=o["epoch"], last_iteration=o["iteration"])
        return emit("hold_static")
    if not 0 < count < config.n:
        if not config.counts(True):
            return emit("unsupported_partition")
        return emit("needs_interior_probe", .5, config.nearest(.5, True))
    if state["eligible"] >= config.budget:
        return emit("budget_exhausted")
    phase_change = state["epoch"] is not None and state["epoch"] != o["epoch"]
    after.update(epoch=o["epoch"], last_iteration=o["iteration"], eligible=state["eligible"]+1)
    if phase_change and config.policy in ("bracketed_bisection", "robust_proportional_v1"):
        after.update(bracket=None, frozen=None, filter=None, dwell=0, phase_hits=0)
    difference = a-b
    sign = (difference > 0) - (difference < 0)
    if config.policy == "bracketed_bisection" and after["frozen"] in ("stop_bracket_width", "stop_resolution", "stop_exact_root"):
        bracket = after["bracket"]
        inconsistent = (after["frozen"] == "stop_exact_root" and sign != 0) or (
            bracket is not None and any((c <= count and value > 0 and difference < 0)
                                       or (c >= count and value < 0 and difference > 0)
                                       for c,value in bracket["samples"]))
        if inconsistent:
            after["frozen"] = "invalid_bracket"
    if after["frozen"]:
        return emit(after["frozen"])
    previous = after["previous_ratio"]
    after["previous_ratio"] = ratio

    if config.policy == "bracketed_bisection":
        legal = config.counts(True)
        if not sign:
            after["frozen"] = "stop_exact_root"
            return emit("stop_exact_root")
        bracket = after["bracket"]
        if bracket is None:
            if count != legal[0]:
                return emit("probe_bracket_low", legal[0]/config.n, legal[0])
            after["bracket"] = dict(lo=count, dlo=difference, hi=None, dhi=None,
                                    samples=[[count, difference]], updates=0)
            return emit("probe_bracket_high", legal[-1]/config.n, legal[-1])
        lo, hi = bracket["lo"], bracket["hi"]
        if hi is None:
            if count != legal[-1]:
                return emit("probe_bracket_high", legal[-1]/config.n, legal[-1])
            bracket.update(hi=count, dhi=difference)
            bracket["samples"].append([count, difference])
            if not (lo < count and bracket["dlo"] < 0 < difference):
                after["frozen"] = "no_bracket"
                return emit("no_bracket")
        else:
            # Every newly observed sign must be consistent with all earlier signs.
            inconsistent = any((c <= count and value > 0 and difference < 0)
                               or (c >= count and value < 0 and difference > 0)
                               for c, value in bracket["samples"])
            if inconsistent or not lo < count < hi:
                after["frozen"] = "invalid_bracket"
                return emit("invalid_bracket")
            bracket["samples"].append([count, difference])
            bracket["updates"] += 1
            if sign > 0:
                bracket.update(hi=count, dhi=difference)
            else:
                bracket.update(lo=count, dlo=difference)
        lo, hi = bracket["lo"], bracket["hi"]
        reason = None
        if (hi-lo)/config.n <= config.precision:
            reason = "stop_bracket_width"
        left, right = bisect_left(legal, lo+1), bisect_left(legal, hi)
        if not reason and left == right:
            reason = "stop_resolution"
        if reason:
            best = min(((lo, abs(bracket["dlo"])), (hi, abs(bracket["dhi"]))), key=lambda x: (x[1], x[0]))[0]
            after["frozen"] = reason
            return emit(reason, best/config.n, best)
        midpoint = (lo+hi)/(2*config.n)
        candidate = config.nearest(midpoint, True)
        # For a sparse grid the nearest midpoint may be an endpoint.
        candidate = max(legal[left], min(legal[right-1], candidate))
        return emit("bisect", midpoint, candidate)

    if config.policy == "paper_binary_literal" and previous is not None and 0 < abs(ratio-previous) < config.precision:
        after["frozen"] = "stop_small_change"
        return emit("stop_small_change")
    if config.policy == "robust_proportional_v1":
        capacity = [a/ratio, b/(1-ratio)]  # Full-work time estimates, not raw times across changing shares.
        if not all(math.isfinite(v) for v in capacity):
            after = deepcopy(before)
            return emit("hold_invalid_observation")
        filtered = after["filter"]
        if filtered is not None:
            jump = any(max(x/y, y/x) >= config.phase_factor for x, y in zip(capacity, filtered))
            after["phase_hits"] = after["phase_hits"]+1 if jump else 0
            if after["phase_hits"] >= config.phase_patience:
                filtered = None
                after.update(dwell=0, phase_hits=0)
                phase_change = True
        after["filter"] = capacity if filtered is None else [config.alpha*x+(1-config.alpha)*y for x,y in zip(capacity,filtered)]
        if abs(difference)/max(a,b) <= config.deadband:
            after["dwell"] = max(0, after["dwell"]-1)
            return emit("hold_deadband")
        if after["dwell"]:
            after["dwell"] -= 1
            return emit("hold_dwell")
        ca, cb = after["filter"]
        scale = max(ca, cb)
        raw = (cb/scale)/(ca/scale+cb/scale)
        move = max(-config.max_move, min(config.max_move, config.damping*(raw-ratio)))
        target = ratio+move
        after["dwell"] = config.dwell-1
        reason = "reset_phase_and_adjust" if phase_change else "robust_adjust"
    elif not sign:
        return emit("hold_equal_time")
    elif config.policy == "paper_proportional":
        # Scaling preserves the equation while avoiding overflow of its denominator.
        scale = max(a,b)
        target = (ratio*(b/scale))/(ratio*(b/scale)+(1-ratio)*(a/scale))
        reason = "proportional_adjust"
    elif config.policy == "paper_binary_literal":
        target = ratio/2 if sign > 0 else (1+ratio)/2
        reason = "literal_binary_adjust"
    elif config.policy == "paper_adaptive":
        step = config.adaptive_step if after["step"] is None else after["step"]
        if after["sign"] and after["sign"] != sign:
            step /= 2
        after.update(step=step, sign=sign)
        if step < config.precision:
            after["frozen"] = "stop_precision"
            return emit("stop_precision")
        target = max(0., min(1., ratio-sign*step))
        reason = "adaptive_adjust"
    else:
        target = max(0., min(1., ratio-sign*config.step))
        realized = config.nearest(target)/config.n
        predicted = a*realized/ratio - b*(1-realized)/(1-ratio)
        if predicted*sign < 0:
            return emit("hold_predicted_oscillation")
        reason = "simple_adjust"
    candidate = config.nearest(target)
    return emit("hold_quantized" if candidate == count else reason, target, candidate)


def balance_trace(observations, tolerance=.05, consecutive=3):
    """Diagnostics never stop or otherwise alter a literal controller."""
    streak = 0
    epoch = None
    seen = -1
    values, balanced = [], []
    for o in observations:
        a,b = o.get("t_cpu"),o.get("t_gpu")
        valid = (o.get("valid") is True and o.get("common_origin") is True
                 and o["iteration"] > seen and all(type(v) in (int,float) and math.isfinite(v) and v > 0 for v in (a,b)))
        if not valid or epoch != o["epoch"]:
            streak = 0
        imbalance = abs(a-b)/max(a,b) if valid else None
        streak = streak+1 if imbalance is not None and imbalance <= tolerance else 0
        values.append(imbalance)
        if streak >= consecutive:
            balanced.append(o["iteration"])
        epoch,seen = o["epoch"],max(seen,o["iteration"])
    return dict(imbalance=values, sustained_at=balanced,
                first_sustained=balanced[0] if balanced else None,
                balanced_at_end=bool(balanced and balanced[-1] == observations[-1]["iteration"]))
