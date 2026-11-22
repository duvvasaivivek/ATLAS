"""Matched guarded clock experiments with independent static tuning and replay."""

from dataclasses import asdict
import json
from pathlib import Path
import random
import shutil
import statistics
import struct
import time

from .actuation import ActuationBlocked
from .capabilities import cpu_interfaces
from .energy import finite_nonnegative
from .experiment import distribution, save_new
from .frequency_experiment import sha
from .integration import Config
from .live import energy_contract, preflight, replay as replay_live, run_live
from .live_native import NativeWorkload
from .performance import interval


def config(plan, trial):
    return Config(trial['id'], plan['items'], trial['cpu_items'], plan['pairs'],
                  allocation_policy=trial['allocation'])


def trial_plan(plan, stage, candidates, blocks):
    rng = random.Random(plan['seed'] + (stage == 'held_out'))
    trials = []
    for block in range(blocks):
        order = list(candidates)
        rng.shuffle(order)
        for name, cpu_items, pair, allocation, change_clocks in order:
            trials.append(dict(id=f'{stage}-{len(trials):04d}', arm=name, block=block,
                               cpu_items=cpu_items, pair=list(pair), allocation=allocation,
                               change_clocks=change_clocks))
    return trials


def tuning_plan(plan):
    cuts = sorted({plan['items']*fraction//4 for fraction in range(4)})
    candidates = [(f'{pair[0]}-{pair[1]}-{cut}', cut, pair, 'static', False)
                  for pair in plan['pairs'] for cut in cuts]
    return trial_plan(plan, 'tuning', candidates, 2)


def select_static(rows):
    if not rows or any(row['status'] != 'passed' for row in rows):
        return None
    scores = {}
    for row in rows:
        scores.setdefault(row['arm'], []).append(row['joules'])
    best = min(scores, key=lambda arm: (statistics.mean(scores[arm]), arm))
    return next(dict(cpu_items=row['cpu_items'], pair=row['pair']) for row in rows if row['arm'] == best)


def evaluation_plan(plan, selection):
    high = plan['high_pair']
    cut = plan['initial_cpu_items']
    arms = [('B0', 0, high, 'static', False),
            ('B2', plan['items']//2, high, 'static', False),
            ('B3', cut, high, 'paper_simple', False),
            ('B4', 0, high, 'static', True),
            ('B5', cut, high, 'paper_simple', True),
            ('B6', selection['cpu_items'], selection['pair'], 'static', False)]
    return trial_plan(plan, 'held_out', arms, plan['blocks'])


def assess_trial(root, plan, trial, dataset):
    directory = root/trial['id']
    if not (directory/'result.json').is_file():
        failure = root/(trial['id']+'.failure.json')
        reason = json.loads(failure.read_text())['reason'] if failure.is_file() else 'Acquisition stopped before trial'
        return dict(trial, status='failed', reason=reason)
    result = replay_live(directory)
    manifest = json.loads((directory/'configuration.json').read_text())
    native = json.loads((directory/'native/plan.json').read_text())
    expected_config = asdict(config(plan, trial))
    expected_config['pairs'] = [list(pair) for pair in expected_config['pairs']]
    qualification = manifest['qualification']
    if (manifest['config'] != expected_config or manifest['initial_pair'] != trial['pair'] or
            manifest['frequency_control'] != trial['change_clocks'] or manifest['measurement'] != plan['meter'] or
            any(qualification[key] != plan[key] for key in ('device_uuid', 'pci_bus_id', 'evidence_id')) or
            native['metadata']['sha256'] != plan['datasets'][dataset] or native['binary_sha256'] != plan['binary_sha256'] or
            any(native['case'][key] != plan[key] for key in ('workload', 'steps', 'workers'))):
        raise ValueError('Live trial differs from its frozen evaluation plan')
    if plan['workload'] == 'kmeans' and (native['case'].get('reduction'), native['case'].get('transfer_mode')) != ('hierarchical', 'streamed'):
        raise ValueError('Controlled k-means trial did not use the matched kernel and transfer mode')
    if result['status'] != 'passed':
        return dict(trial, status='failed', reason=result['status'])
    energy = result['energy']
    if not energy['physical_measurement'] or not finite_nonnegative(energy['joules']) or energy['joules'] == 0:
        return dict(trial, status='failed', reason=energy.get('reason', 'No positive qualified physical energy'))
    return dict(trial, status='passed', joules=energy['joules'],
                meter_accuracy_j=energy['meter_accuracy_j'],
                measured_s=(energy['end_ns']-energy['start_ns'])/1e9,
                lifecycle_s=result['elapsed_ns']/1e9)


def paired_comparisons(rows):
    comparisons = []
    for arm, reference in (('B2','B0'), ('B3','B0'), ('B4','B0'), ('B5','B0'), ('B5','B3'), ('B6','B0')):
        baseline = {row['block']: row for row in rows if row['arm'] == reference and row['status'] == 'passed'}
        requested = [row for row in rows if row['arm'] == arm]
        pairs = [(row, baseline[row['block']]) for row in requested if row['status'] == 'passed' and row['block'] in baseline]
        energy = [100*(1-row['joules']/base['joules']) for row,base in pairs]
        runtime = [100*(row['measured_s']/base['measured_s']-1) for row,base in pairs]
        comparisons.append(dict(arm=arm, reference=reference, requested=len(requested), pairs=len(pairs),
                                energy_saving_percent=distribution(energy), energy_bootstrap_95_percent=interval(energy) if len(pairs)>1 else None,
                                runtime_change_percent=distribution(runtime), runtime_bootstrap_95_percent=interval(runtime) if len(pairs)>1 else None))
    return comparisons


def summarize(root):
    plan = json.loads((root/'plan.json').read_text())
    tuning = [assess_trial(root, plan, trial, 'tuning.data') for trial in tuning_plan(plan)]
    selection = select_static(tuning)
    rows = []
    if selection is not None:
        frozen = evaluation_plan(plan, selection)
        if frozen != json.loads((root/'held_out_plan.json').read_text()):
            raise ValueError('Held-out plan differs from independent tuning evidence')
        rows = [assess_trial(root, plan, trial, 'held_out.data') for trial in frozen]
    complete = bool(rows) and all(row['status'] == 'passed' for row in rows)
    return dict(status='passed' if complete else 'incomplete', domain=plan['meter']['domain'],
                scope='first native release through completion and final output verification',
                failed_trials=sum(row['status'] != 'passed' for row in tuning+rows),
                selection=selection, tuning=tuning, trials=rows, comparisons=paired_comparisons(rows) if rows else [])


def archive_files(root):
    return {path.relative_to(root).as_posix(): path for path in root.rglob('*') if path.is_file() and path != root/'checksums.json'}


def replay(root):
    root = Path(root)
    files = archive_files(root)
    hashes = json.loads((root/'checksums.json').read_text())
    if files.keys() != hashes.keys() or any(path.is_symlink() or sha(path) != hashes[name] for name,path in files.items()):
        raise ValueError('Controlled evaluation archive integrity failure')
    plan = json.loads((root/'plan.json').read_text())
    if any(hashes[name] != plan['datasets'][name] for name in ('tuning.data', 'held_out.data')):
        raise ValueError('Controlled evaluation dataset identity mismatch')
    result = summarize(root)
    if result != json.loads((root/'summary.json').read_text()):
        raise ValueError('Controlled evaluation replay mismatch')
    return result


def run(provider, output, binary, workload, tuning, held_out, high_pair, pairs, *, steps, workers, blocks, initial_fraction):
    budgets = ((workers, 1, 8), (steps, 1, 100000), (blocks, 2, 30))
    if workload not in ('kmeans','hotspot') or any(type(value) is not int or not lo <= value <= hi for value,lo,hi in budgets):
        raise ValueError('Invalid workload, iteration, worker, or block count')
    if initial_fraction not in (.15,.5,.9):
        raise ValueError('Initial CPU fraction must be 0.15, 0.5, or 0.9')
    high_pair = tuple(high_pair)
    candidates = {tuple(pair) for pair in pairs} | {high_pair}
    if not 2 <= len(candidates) <= 8:
        raise ValueError('Choose 2 to 8 qualified clock pairs including the fixed high pair')
    binary = Path(binary).resolve(strict=True)
    inputs = {'tuning.data': Path(tuning), 'held_out.data': Path(held_out)}
    magic = b'ATLSKM01' if workload == 'kmeans' else b'ATLSHS01'
    shapes = []
    for source in inputs.values():
        with source.open('rb') as stream:
            header = stream.read(20 if workload == 'kmeans' else 16)
        if header[:8] != magic or len(header) != (20 if workload == 'kmeans' else 16):
            raise ValueError('Unexpected tuning or held-out workload input')
        shapes.append(struct.unpack('<'+'I'*((len(header)-8)//4), header[8:]))
    hashes = {name: sha(source) for name,source in inputs.items()}
    if shapes[0] != shapes[1] or hashes['tuning.data'] == hashes['held_out.data'] or shapes[0][0] < 2:
        raise ValueError('Use different tuning and held-out inputs of the same shape with mixed work available')
    items = shapes[0][0]
    probe = Config('controlled-evaluation', items, 0, tuple(sorted(candidates)), allocation_policy='static')
    qualification = preflight(provider, probe, time.monotonic_ns())
    if qualification['evidence_kind'] != 'hardware':
        raise ActuationBlocked('Controlled evaluation requires qualified physical control')
    meter = energy_contract(provider, qualification)
    plan = dict(workload=workload, items=items, initial_cpu_items=int(items*initial_fraction),
                pairs=[list(pair) for pair in probe.pairs], high_pair=list(high_pair), steps=steps, workers=workers,
                blocks=blocks, seed=20261122, datasets=hashes, binary_sha256=sha(binary),
                device_uuid=qualification['device_uuid'], pci_bus_id=qualification['pci_bus_id'],
                evidence_id=qualification['evidence_id'], meter=meter, cpu_policy=cpu_interfaces()['governors'])
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    for name,source in inputs.items():
        shutil.copyfile(source, root/name)
        if sha(root/name) != hashes[name]:
            raise ValueError('Benchmark input changed before capture')
    save_new(root/'plan.json', plan)

    def collect(trials, dataset):
        for trial in trials:
            directory = root/trial['id']
            native = NativeWorkload(binary, root/dataset, workload, steps, workers, directory/'native', optimized_kmeans=True)
            try:
                run_live(provider, native, config(plan, trial), directory, initial_pair=trial['pair'],
                         frequency_control=trial['change_clocks'], measure=True)
            except (RuntimeError, OSError, ValueError) as error:
                # A severe failure ends acquisition. Do not start another clock transaction.
                save_new(root/(trial['id']+'.failure.json'), dict(reason=f'{type(error).__name__}: {error}'))
                break

    pilot_trials = tuning_plan(plan)
    collect(pilot_trials, 'tuning.data')
    pilot = [assess_trial(root, plan, trial, 'tuning.data') for trial in pilot_trials]
    selection = select_static(pilot)
    if selection is not None:
        trials = evaluation_plan(plan, selection)
        save_new(root/'held_out_plan.json', trials)
        collect(trials, 'held_out.data')
    result = summarize(root)
    save_new(root/'summary.json', result)
    save_new(root/'checksums.json', {name: sha(path) for name,path in archive_files(root).items()})
    return replay(root)
