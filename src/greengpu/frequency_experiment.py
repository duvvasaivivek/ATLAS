"""Compact offline evidence, read-only traces, and deterministic decision replay."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys

from .experiment import encode, save_new
from .frequency import Config, SPEC_VERSION, initial_state, transition
from .frequency_scenarios import definitions, simulate, summarize
from .frequency_trace import capture, pair_samples


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_lines(path, records):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for record in records: stream.write(encode(record)+"\n")


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def decisions_for(config, run_id, observations):
    state = initial_state(config,run_id)
    decisions = []
    for observation in observations:
        decision = transition(config,state,observation)
        decisions.append(decision); state = decision["post_state"]
    return decisions


def evaluate(root):
    manifest = json.loads((root/"manifest.json").read_text())
    if manifest["spec_version"] != SPEC_VERSION:
        raise ValueError("Unsupported frequency specification")
    if encode(manifest["synthetic_cases"]) != encode(definitions()):
        raise ValueError("Diagnostic matrix differs from the versioned specification")
    summaries = {}
    beta_choices = []
    for case in manifest["synthetic_cases"]:
        obs, expected = simulate(case)
        retained_obs = read_lines(root/(case["run_id"]+".observations.jsonl"))
        retained_decisions = read_lines(root/(case["run_id"]+".decisions.jsonl"))
        if retained_obs != obs or retained_decisions != expected:
            raise ValueError("Synthetic trace/decision mismatch: "+case["run_id"])
        summaries[case["run_id"]] = summarize(Config(**case["config"]),expected)
        if case["run_id"].startswith("beta-"):
            beta_choices.append([d["recommended_pair"] for d in expected])
    if not all(choices == beta_choices[0] for choices in beta_choices):
        raise ValueError("Beta diagnostic rankings differ")
    hardware = dict(status="not_requested", observations=0, updates=0)
    if manifest["hardware_config"] is not None:
        raw = json.loads((root/"hardware.json").read_text())
        config = Config(**manifest["hardware_config"])
        if config.pairs != tuple(sorted(tuple(p) for p in raw["selected_pairs"])):
            raise ValueError("Hardware candidates do not match retained discovery")
        if any(list(p) not in raw["reported_pairs"] for p in config.pairs):
            raise ValueError("Unreported hardware candidate")
        obs = pair_samples(raw)
        expected = decisions_for(config,"hardware-offline",obs)
        if read_lines(root/"hardware.observations.jsonl") != obs or read_lines(root/"hardware.decisions.jsonl") != expected:
            raise ValueError("Hardware trace/decision mismatch")
        hardware = summarize(config,expected)
        hardware.update(status="replayed_offline" if hardware["updates"] else "blocked_no_eligible_observations_or_candidates",
                        reported_pairs=len(raw["reported_pairs"]),selected_pairs=len(config.pairs),
                        window_contract=raw["eligibility"], workload=raw["workload"],
                        integration_bounds_qualified=False, actuation_verified=False)
    return dict(spec_version=SPEC_VERSION, implementation_status="passed",
                synthetic_cases=summaries, synthetic_observations=sum(s["observations"] for s in summaries.values()),
                beta_diagnostic=dict(values=[.01,.2,.99], identical_selections=True,
                                     conclusion="fixed_beta_does_not_forget_history"),
                hardware=hardware, offline_gate="passed", live_actuation="not_attempted_phase8",
                energy_savings="not_evaluated", hardware_counterfactuals=False,
                prior_phase_traces="Phases 2-6 lack paired memory utilization; not eligible for Tier 2 replay")


def run(output, hardware=False):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    cases = definitions()
    for case in cases:
        observations, decisions = simulate(case)
        write_lines(output/(case["run_id"]+".observations.jsonl"),observations)
        write_lines(output/(case["run_id"]+".decisions.jsonl"),decisions)
    hardware_config = None
    if hardware:
        raw = capture()
        config = Config(raw["selected_pairs"],provenance="hardware_reported_unverified")
        hardware_config = asdict(config)
        obs = pair_samples(raw)
        save_new(output/"hardware.json",raw)
        write_lines(output/"hardware.observations.jsonl",obs)
        write_lines(output/"hardware.decisions.jsonl",decisions_for(config,"hardware-offline",obs))
    save_new(output/"manifest.json",dict(spec_version=SPEC_VERSION,synthetic_cases=cases,hardware_config=hardware_config,
                                        python=sys.version,platform=platform.platform(),hardware_settings_changed=False))
    paths = [p for folder in ("src","tests","native") for p in Path(folder).rglob("*")
             if p.is_file() and p.suffix in (".py",".cpp",".cu",".hpp",".h")]
    paths.extend(Path(p) for p in ("pyproject.toml","CMakeLists.txt"))
    save_new(output/"source_hashes.json",{p.as_posix():sha(p) for p in sorted(paths)})
    summary = evaluate(output)
    save_new(output/"summary.json",summary)
    save_new(output/"checksums.json",{p.name:sha(p) for p in sorted(output.iterdir())})
    replay(output)
    return summary


def replay(root):
    root = Path(root)
    hashes = json.loads((root/"checksums.json").read_text())
    inventory = {p.name for p in root.iterdir() if p.name != "checksums.json"}
    if set(hashes) != inventory:
        raise ValueError("Archive inventory mismatch")
    for name, expected in hashes.items():
        path = root/name
        if path.is_symlink() or not path.is_file() or sha(path) != expected:
            raise ValueError("Archive checksum mismatch: "+name)
    result = evaluate(root)
    if result != json.loads((root/"summary.json").read_text()):
        raise ValueError("Summary replay mismatch")
    return result
