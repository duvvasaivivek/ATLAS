"""Regenerate a capability-limited research report from verified raw archives.

This command never measures hardware or promotes shadow decisions to actuation.
The HTML is standalone; the JSON retains estimates, source hashes and blockers.
"""
import html
import json
from pathlib import Path
import statistics

from .experiment import save_new
from .frequency_experiment import sha
from .performance import interval


def command_comparisons(results):
    """Pair independent blocks, including only successful matched commands."""
    comparisons = []
    groups = sorted({(r['workload'], r['size']) for r in results
                     if r['stage'] == 'evaluation'})
    for workload, size in groups:
        rows = [r for r in results if (r['workload'], r['size'], r['stage']) ==
                (workload, size, 'evaluation')]
        baseline = {r['block']: r['command_s'] for r in rows
                    if r['name'] == 'gpu_only' and r['status'] == 'passed'}
        for name in sorted({r['name'] for r in rows}):
            requested = [r for r in rows if r['name'] == name]
            paired = [r for r in requested if r['status'] == 'passed'
                      and r['block'] in baseline]
            if any(baseline[r['block']] <= 0 or r['command_s'] <= 0 for r in paired):
                raise ValueError('Nonpositive command duration')
            deltas = [100 * (r['command_s'] / baseline[r['block']] - 1) for r in paired]
            comparisons.append(dict(workload=workload, size=size, name=name,
                                    pairs=len(paired), unpaired=len(requested)-len(paired),
                                    change_percent=statistics.mean(deltas) if deltas else None,
                                    bootstrap_95_percent=interval(deltas)))
    return comparisons


def assemble(performance, integration, qualification, plan):
    """Separate artifact generation, correctness, acceptance and physical claims."""
    blockers = []
    if performance['status'] != 'passed':
        blockers.append('Performance evaluation contains failed trials.')
    required = {'gpu_only', 'cpu_only', 'static50', 'tuned_static', 'cost_aware'}
    matrix_complete = True
    for workload in ('kmeans', 'hotspot'):
        for size in (0, 1):
            rows = [r for r in performance['results'] if
                    (r['workload'], r['size'], r['stage']) == (workload, size, 'evaluation')]
            for name in required:
                blocks = [r['block'] for r in rows if r['name'] == name and r['status'] == 'passed']
                if len(blocks) != plan['blocks'] or set(blocks) != set(range(plan['blocks'])):
                    matrix_complete = False
    if not matrix_complete:
        blockers.append('The primary allocation evaluation matrix is incomplete.')
    if plan['blocks'] < 10:
        blockers.append('Fewer than the planned ten independent evaluation blocks were collected.')
    if integration['software_gate'] != 'passed':
        blockers.append('Integrated software acceptance: ' + integration['software_gate'] + '.')
    if qualification['actuation_gate'] != 'passed':
        blockers.append('Authoritative GPU clock-policy readback and verified actuation/restoration are unavailable.')
    if not qualification['clock_mapping']['live_qualified']:
        blockers.append('Driver telemetry timestamps are not qualified for live feedback.')
    if qualification['device_timing_qualification'] != 'qualified':
        blockers.append('CUDA device-event timing requires independent qualification.')
    if qualification['energy_qualification'] != 'qualified':
        blockers.append('Physical energy measurement is unqualified; energy savings are unavailable.')
    # These archive contracts are explicitly allocation/shadow + automatic clocks.
    # A successful report must never imply that missing B0/B3/B4/B5/B6 ran.
    blockers.append('Fixed-clock allocation, frequency-only, live joint-control and joint static-sweep baselines have not been evaluated.')
    concurrency = {name: dict(status=value['status'],
                             measured_iterations=len(value['iterations']),
                             overlapping_iterations=sum(r['demonstrated'] for r in value['iterations']))
                   for name, value in integration['concurrency'].items()}
    selections = []
    for key, count in sorted(plan['selections'].items()):
        workload, size = key.rsplit('-', 1)
        dataset = plan['datasets'][f'datasets/{workload}-{size}-evaluation.data']
        total = dataset['n'] if workload == 'kmeans' else dataset['rows']
        selections.append(dict(workload=workload, size=int(size), cpu_items=count,
                               total_items=total, cpu_fraction=count/total))
    return dict(schema_version='research-report-1.0', full_project_complete=False,
                scope='allocation-only hardware evaluation and frequency shadow/simulation',
                status='capability_limited', blockers=blockers,
                evaluation=dict(status=performance['status'], trials=performance['trials'],
                                failed_trials=performance['failed_trials'], blocks=plan['blocks'],
                                primary_matrix_complete=matrix_complete,
                                primary_scope=performance['primary_scope'],
                                baseline=performance['baseline'], selections=selections,
                                workload_comparisons=performance['comparisons'],
                                command_comparisons=command_comparisons(performance['results'])),
                integration=dict(software_gate=integration['software_gate'],
                                 live_hardware_gate=integration['live_joint_hardware_gate'],
                                 hardware_trials=integration['hardware_trials'],
                                 allocation_decisions=integration['allocation_decisions'],
                                 ownership_changes=integration['ownership_changes'],
                                 concurrency=concurrency),
                qualification=qualification, energy_savings_percent=None,
                statistical_scope='Pointwise exploratory 95% paired bootstrap intervals; 2000 fixed-seed resamples; no multiple-comparison adjustment.',
                limitations=['Automatic GPU clocks; fixed hardware capacity is not established.',
                             'No thermal isolation or exclusive host scheduling is established.',
                             'Generated inputs are not the original paper datasets.',
                             'Workload timing excludes final validation; full command timing includes it.',
                             'Tuning selects a static partition on separate inputs; no learned model is trained.',
                             'Positive, negative and inconclusive outcomes are retained.'])


def render(report):
    escape = lambda value: html.escape(str(value), quote=True)
    def estimate(value, bounds):
        if value is None:
            return 'Unavailable'
        return f'{value:+.1f}% [{bounds[0]:+.1f}%, {bounds[1]:+.1f}%]'
    commands = {(r['workload'], r['size'], r['name']): r
                for r in report['evaluation']['command_comparisons']}
    rows = []
    for row in report['evaluation']['workload_comparisons']:
        command = commands[(row['workload'], row['size'], row['name'])]
        values = [row['workload'], row['size'], row['name'], row['trials'],
                  estimate(row['paired_runtime_change_percent'], row['paired_bootstrap_95_percent']),
                  estimate(command['change_percent'], command['bootstrap_95_percent'])]
        rows.append('<tr>' + ''.join(f'<td>{escape(v)}</td>' for v in values) + '</tr>')
    blockers = ''.join(f'<li>{escape(v)}</li>' for v in report['blockers'])
    limits = ''.join(f'<li>{escape(v)}</li>' for v in report['limitations'])
    sources = ''.join(f'<li>{escape(k)}: {escape(v["path"])}; checksum manifest SHA-256 '
                      f'<code>{escape(v["checksums_sha256"])}</code></li>'
                      for k, v in report['sources'].items())
    selections = ''.join(f'<li>{escape(r["workload"])} size {r["size"]}: '
                         f'{r["cpu_items"]}/{r["total_items"]} CPU items '
                         f'({100*r["cpu_fraction"]:.1f}%)</li>'
                         for r in report['evaluation']['selections'])
    overlap = ''.join(f'<li>{escape(k)}: {v["overlapping_iterations"]}/'
                      f'{v["measured_iterations"]} iterations demonstrate compute overlap; '
                      f'{escape(v["status"])}</li>'
                      for k, v in report['integration']['concurrency'].items())
    return f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>ATLAS reproducible research report</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;max-width:1200px;margin:40px auto;padding:0 24px;color:#18212b;background:#fafafa}}
h1{{font-size:28px}}h2{{font-size:21px;margin-top:32px}}table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{text-align:left;padding:10px;border-bottom:1px solid #ccd2d8}}th{{background:#edf0f3}}code{{overflow-wrap:anywhere}}
.notice{{padding:16px;border-left:4px solid #906b1e;background:#fff6dc}}.scroll{{overflow-x:auto}}</style>
<h1>ATLAS GreenGPU research report</h1>
<p class="notice"><strong>Capability-limited result. Full project acceptance remains incomplete.</strong>
Energy savings and live frequency-control benefits have not been established.</p>
<p>All three source archives passed checksum verification and raw-result replay before report generation.
Evaluation: {report['evaluation']['trials']} trials, {report['evaluation']['failed_trials']} failures,
{report['evaluation']['blocks']} independent held-out blocks per variant.</p>
<h2>Static partition tuning</h2><p>Separate tuning inputs select a partition before held-out evaluation.
Size indices refer to the exact datasets in the source plan.json.</p><ul>{selections}</ul>
<h2>Runtime change versus matched GPU-only</h2>
<p>Negative percentages mean faster. Brackets show pointwise 95% paired bootstrap intervals.
Workload and full-command intervals have different measurement boundaries; a workload speedup does not establish a total-job speedup.</p>
<div class="scroll"><table><thead><tr><th>Workload</th><th>Size index</th><th>Variant</th><th>Matched pairs</th>
<th>Workload change (%) [95% CI]</th><th>Full-command change (%) [95% CI]</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p>{escape(report['statistical_scope'])}</p>
<h2>Integration</h2><p>{report['integration']['allocation_decisions']} allocation decisions;
{report['integration']['ownership_changes']} ownership changes.</p><ul>{overlap}</ul>
<h2>Unfinished acceptance gates</h2><ul>{blockers}</ul>
<h2>Interpretation limits</h2><ul>{limits}</ul>
<h2>Verified sources</h2><ul>{sources}</ul>
<p>The accompanying report.json contains machine-readable estimates, qualifications and source identities.
No network requests, external assets or simulated energy values are used.</p></html>'''


def run(output, performance_path, integration_path, qualification_path):
    from .performance import replay as replay_performance
    from .phase9 import replay as replay_integration
    from .qualification import replay as replay_qualification
    root = Path(output).resolve()
    paths = dict(performance=Path(performance_path).resolve(),
                 integration=Path(integration_path).resolve(),
                 qualification=Path(qualification_path).resolve())
    if any(root == p or root.is_relative_to(p) for p in paths.values()):
        raise ValueError('Report output must be outside immutable evidence archives')
    if root.exists():
        raise FileExistsError('Report output already exists; choose a fresh directory')
    performance = replay_performance(paths['performance'])
    integration = replay_integration(paths['integration'])
    qualification = replay_qualification(paths['qualification'])
    plan = json.loads((paths['performance']/'plan.json').read_text())
    result = assemble(performance, integration, qualification, plan)
    result['sources'] = {name: dict(path=str(path), checksums_sha256=sha(path/'checksums.json'),
                                  summary_sha256=sha(path/'summary.json'))
                         for name, path in paths.items()}
    root.mkdir(parents=True, exist_ok=False)
    save_new(root/'report.json', result)
    (root/'report.html').write_text(render(result), encoding='utf-8')
    save_new(root/'checksums.json', {p.name: sha(p) for p in sorted(root.iterdir())})
    return result
