#!/usr/bin/env python3
"""Small, bounded presentation workload using the existing real native runner."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import html
import json
import os
import platform
import subprocess
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from greengpu import kmeans_reference as kr
from greengpu.integration import Config
from greengpu.integration_runner import execute, replay_decisions
from greengpu.demo_evidence import Monitor, import_points, step_evidence


def write(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def presentation(data):
    """Portable presentation: real measured results, embedded sample, no server."""
    payload = json.dumps(data, allow_nan=False).replace('<', '\\u003c')
    fastest = min(data['modes'], key=lambda row: row['workload_ms'])
    rows = ''.join(f'<tr><th>{html.escape(row["mode"])}</th><td>{row["workload_ms"]:.2f}</td>'
                   f'<td>{row["min_ms"]:.2f}–{row["max_ms"]:.2f}</td><td>{row["command_ms"]:.2f}</td></tr>' for row in data['modes'])
    template = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ATLAS | Can two processors share the job?</title><style>
*{box-sizing:border-box}body{margin:0;background:#f5f7f8;color:#172c32;font:16px/1.55 system-ui,sans-serif}main{max-width:1150px;margin:auto;padding:36px 24px}h1{font-size:34px;line-height:1.2;max-width:850px}h2{font-size:21px;margin-top:0}.eyebrow{color:#176c55;font-weight:700;letter-spacing:.08em;font-size:13px}.intro{font-size:19px;max-width:900px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}.panel{background:white;border:1px solid #dce3e5;padding:24px;margin-top:24px;border-radius:8px}.wide{grid-column:1/-1}.muted,small{color:#53686d}.badge{display:inline-block;border:1px solid #b9cfc7;border-radius:20px;padding:4px 12px;font-size:13px}button{background:#176c55;color:white;border:0;padding:10px 16px;font:inherit;cursor:pointer;border-radius:4px}button.secondary{background:#e6eeeb;color:#174936}svg{display:block;width:100%;height:auto}svg text{font-family:system-ui,sans-serif;font-size:12px;fill:#53686d}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:10px;border-bottom:1px solid #e0e5e6}td{text-align:right}.metric{font-size:27px;font-weight:700;color:#176c55}.legend{display:flex;flex-wrap:wrap;gap:16px;font-size:13px}.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}details{margin-top:18px}summary{cursor:pointer;font-weight:600}.callout{border-left:4px solid #176c55;padding-left:18px}footer{margin:30px 0;color:#53686d;font-size:14px}@media(max-width:760px){.grid{grid-template-columns:1fr}h1{font-size:28px}.panel{padding:16px}table{font-size:12px}th,td{padding:6px}}@media print{button{display:none}.panel{break-inside:avoid}body{background:white}}</style>
<main><div class="eyebrow">ATLAS / A REAL CPU + GPU DEMONSTRATION</div><h1>One job. Two different teams.<br>Is sharing the work worth it?</h1>
<p class="intro">Group <strong id="size"></strong> delivery stops into four nearby areas. A CPU is a small team of versatile workers. A GPU is a large team doing similar calculations. ATLAS adjusts how much work each team receives.</p>
<span class="badge">RECORDED MEASUREMENT · __DATE__</span> <span class="badge">Synthetic locations · real CPU/CUDA execution</span>
<p class="muted">The research problem: using both processors can reduce waiting, but transferring data and coordinating them also takes time. The best split depends on the workload and machine. GreenGPU's broader goal is energy conservation; energy savings have not been established on this setup.</p>
<div class="grid"><section class="panel"><h2>1 / Here is the job</h2><p>Every dot is a delivery stop. The algorithm repeatedly assigns nearby stops to the same area.</p>
<button id="toggle">Show the groups</button><svg id="map" viewBox="0 0 500 330" role="img" aria-label="Sample delivery locations before and after grouping"></svg>
<div class="legend" id="legend"></div><small id="sample"></small><p><small>Coordinates are synthetic east–west / north–south units. This groups locations; it does not plan vehicle routes.</small></p></section>
<section class="panel"><h2>2 / Compare the same job three ways</h2><p>CPU only, GPU only, then ATLAS adjusting the share. Identical input and 32 rounds in every trial.</p>
<svg id="times" viewBox="0 0 500 300" role="img" aria-label="Median measured workload time by execution mode"></svg>
<p class="callout"><strong>Fastest in this demonstration: __FASTEST__.</strong><br>Sharing is useful only when the saved work outweighs coordination and transfer costs.</p><small>Median of three shuffled repetitions. Lower time is better. Bars show workload time, excluding startup, warmup and final independent validation.</small></section>
<section class="panel"><h2>3 / What did ATLAS actually change?</h2><svg id="trace" viewBox="0 0 500 230" role="img" aria-label="Actual CPU workload share over grouping rounds"></svg>
<p><strong id="changes"></strong> allocation changes in the first ATLAS trial. The GPU gets the remaining share: 30% CPU means 70% GPU.</p><small>These are recorded decisions, not a live animation or proof of compute overlap. Allocation timing is measured; CPU/GPU capacity is not assumed constant.</small></section>
<section class="panel"><h2>4 / Did the answer stay correct?</h2><div class="metric">Same groups. All modes.</div><p>Every trial passed a native independent reference check. Final memberships and group centers match across all execution modes and repetitions. Controller decisions were replayed.</p>
<p><strong>What this proves:</strong> this input can be processed correctly while ATLAS changes the CPU/GPU work split.</p><p><strong>What this cannot prove:</strong> general speedup, energy savings, distributed scalability or full GreenGPU acceptance.</p></section>
<section class="panel wide"><h2>Explain it to your classmates in 30 seconds</h2><p class="intro">“We gave the computer a location-grouping job. Instead of always sending everything to the GPU, ATLAS can use both the CPU and GPU and adjust their shares. We compare the same task three ways, check the answers match, and measure the time. Sometimes sharing helps; sometimes coordination costs more. This demo shows that tradeoff on this machine.”</p><p>Grouping locations is the example workload. ATLAS's contribution is deciding which processor does how much of that work. The project's second research workload simulates heat flow; this presentation focuses on the grouping example.</p>
<p><strong>To show scale:</strong> run Small, Medium and Large in the launcher. Compare how the outcome changes as the job grows. These are bounded single-host sizes: 2,000, 16,000 and 64,000 stops. Each run records actual results; larger inputs are not promised to run faster.</p>
<details><summary>Measured times, reliability and scope</summary><div style="overflow:auto"><table><thead><tr><th>Execution mode</th><th>Median workload (ms)</th><th>Min–max (ms)</th><th>Median command (ms)</th></tr></thead><tbody>__ROWS__</tbody></table></div>
<p>Command time includes process startup, warmup and final reference validation. Both timing scopes are shown so that moving work outside the measured loop cannot hide total cost. Live observation and snapshot updates also add overhead; these are monitored demo timings. Three repetitions make this a demonstration, not a statistical qualification.</p>
<p>Fixed seed 1729, four CPU workers, 32 grouping rounds, automatic clocks. Each native trial has a 90-second deadline. No automatic retry hides failures. Raw inputs, controller records, checksums and build identity accompany this page.</p>
<p>Current project status: capability-limited. Authoritative GPU clock control, physical energy measurement, telemetry/device timing qualification and strict all-iteration compute overlap remain incomplete.</p></details></section></div>
<section class="panel"><h2>Where did a row go?</h2>
<p id="origin"></p><label>Try <select id="trial"></select></label>
<label>Step <select id="step"></select></label>
<label>Input row <input id="row" type="number" min="1" value="1"></label>
<p id="placement" aria-live="polite"></p><p id="decision"></p><p id="timing"></p><p id="transfers"></p>
<details><summary>Technical details</summary><pre id="technical"></pre></details>
<p>Input row numbers match the exported <a href="groups.csv">groups.csv</a> when opened from the run folder.</p>
</section>
<section class="panel"><h2>What did this computer report?</h2>
<p id="machine"></p><label id="sample-label">Recorded reading <select id="machine-pick"></select></label>
<p id="power"></p><p id="monitoring"></p>
<small>Linux CPU readings cover the Linux environment. Graphics readings cover the whole device and may include other applications. Power is a sensor reading in watts. Energy savings require a separate qualified experiment.</small>
</section>
<footer>This recorded report opens offline. Use the launcher to measure a new run.</footer></main>
<script>const d=__DATA__,ns='http://www.w3.org/2000/svg',colors=['#377bb5','#c57427','#328460','#8662a8'];function el(tag,attrs,text){let e=document.createElementNS(ns,tag);Object.entries(attrs||{}).forEach(([k,v])=>e.setAttribute(k,v));if(text!==undefined)e.textContent=text;return e}function text(svg,x,y,s,extra={}){svg.append(el('text',{x,y,...extra},s))}document.querySelector('#size').textContent=d.size.toLocaleString();document.querySelector('#sample').textContent=`Showing ${d.points.length} sampled stops; all ${d.size.toLocaleString()} stops were processed.`;document.querySelector('#changes').textContent=d.allocation_changes;
let grouped=false;function map(){const svg=document.querySelector('#map');svg.replaceChildren();text(svg,250,325,'East–west coordinate (synthetic units)',{'text-anchor':'middle'});for(let i=0;i<d.points.length;i++){let [x,y]=d.points[i];svg.append(el('circle',{cx:25+(x+10)/120*450,cy:295-(y+10)/120*270,r:3,fill:grouped?colors[d.labels[i]]:'#92a1a6',opacity:.8}))}document.querySelector('#legend').replaceChildren();if(grouped){d.counts.forEach((n,i)=>{let span=document.createElement('span'),dot=document.createElement('i');dot.className='dot';dot.style.background=colors[i];span.append(dot,`Area ${i+1}: ${n.toLocaleString()} stops`);document.querySelector('#legend').append(span)})}document.querySelector('#toggle').textContent=grouped?'Show original locations':'Show the groups'}document.querySelector('#toggle').onclick=()=>{grouped=!grouped;map()};map();
const bars=document.querySelector('#times'),max=Math.max(...d.modes.map(m=>m.workload_ms))*1.12;d.modes.forEach((m,i)=>{let y=30+i*75,w=m.workload_ms/max*300;text(bars,5,y+20,m.mode);bars.append(el('rect',{x:145,y,width:w,height:32,fill:i===2?'#176c55':'#8b9fa7'}));text(bars,150+w,y+21,m.workload_ms.toFixed(2)+' ms')});text(bars,250,278,'Median workload time (milliseconds) · lower is better',{'text-anchor':'middle'});
const tr=document.querySelector('#trace');for(let p of [0,50,100]){let y=185-p*1.5;tr.append(el('line',{x1:42,y1:y,x2:478,y2:y,stroke:'#e1e7e9'}));text(tr,35,y+4,p+'%',{'text-anchor':'end'})}tr.append(el('polyline',{points:d.trace.map(r=>`${42+(r.round-1)/31*436},${185-r.cpu_percent*1.5}`).join(' '),fill:'none',stroke:'#176c55','stroke-width':3}));text(tr,250,221,'Grouping round 1 → 32 (first ATLAS trial)',{'text-anchor':'middle'});text(tr,45,20,'CPU share (%) · GPU receives the rest');</script></html>'''
    inspector = """
const get = id => document.getElementById(id);
const fill = (select, labels) => select.replaceChildren(...labels.map((label, i) => new Option(label, i)));
const fmt = (v, unit) => v == null ? 'Unavailable' : v.toFixed(2) + ' ' + unit;
const trial = get('trial'), step = get('step'), row = get('row');
const modes = {'CPU only':'Main processor alone', 'GPU only':'Graphics processor alone', 'ATLAS sharing':'Both working together'};
const reasons = {
  collect_epoch:'Collecting timings before the next decision.',
  collect_native_batch:'Collecting timings before the next decision.',
  hold_static:'This comparison uses a fixed assignment.',
  hold_equal_time:'Both shares finished in the same measured time.',
  hold_quantized:'The nearest allowed split stayed the same.',
  hold_invalid_observation:'The controller held the split because the timing was unusable.',
  proportional_adjust:'ATLAS used both completion times to calculate the next split.',
  workload_complete:'This was the last step.'
};
get('origin').textContent = d.synthetic ? 'Input: generated sample locations (repeatable seed 1729).' : 'Input: imported CSV, ' + d.input.name;
row.max = d.size;
fill(trial, d.step_trials.map(t => modes[t.mode] + ' · ' + t.trial));
function inspect() {
  const t = d.step_trials[Number(trial.value)], s = t.steps[Number(step.value)];
  const r = Number(row.value);
  const valid = Number.isInteger(r) && r >= 1 && r <= d.size;
  const cpu = s.cpu_items ? 'main processor handled rows 1–' + s.cpu_items : 'main processor handled no rows';
  const gpu = s.gpu_items ? 'graphics processor handled rows ' + (s.cpu_items + 1) + '–' + d.size : 'graphics processor handled no rows';
  get('placement').textContent = (valid ? 'Row ' + r + ' went to the ' + (r <= s.cpu_items ? 'main processor. ' : 'graphics processor. ') : 'Choose a row from 1 to ' + d.size + '. ') + 'This step: ' + cpu + '; ' + gpu + '.';
  const next = s.decision === 'workload_complete' || s.next_cpu_items == null ? '' : ' Next step: ' + s.next_cpu_items + ' rows assigned to the main processor.';
  get('decision').textContent = (reasons[s.decision] || (s.decision ? 'See the recorded decision below.' : 'Decision unavailable.')) + next;
  const start = t.steps[0].release_monotonic_ns;
  const offset = start == null || s.release_monotonic_ns == null ? null : (s.release_monotonic_ns - start) / 1e6;
  get('timing').textContent = 'Started ' + fmt(offset, 'ms') + ' after the first step · Step duration: ' + fmt(s.step_ms, 'ms') + ' · Main processor work: ' + fmt(s.cpu_work_ms, 'ms') + ' · Graphics dispatch to returned results: ' + fmt(s.gpu_host_ms, 'ms') + ' · Combine results: ' + fmt(s.merge_ms, 'ms');
  const upload = s.input_upload_bytes == null || s.center_upload_bytes == null ? null : s.input_upload_bytes + s.center_upload_bytes;
  const download = s.labels_download_bytes == null || s.aggregates_download_bytes == null ? null : s.labels_download_bytes + s.aggregates_download_bytes;
  get('transfers').textContent = 'Sent to the graphics processor: ' + fmt(upload, 'bytes') + ' · Returned from the graphics processor: ' + fmt(download, 'bytes');
  get('technical').textContent = JSON.stringify({linux_pid:t.linux_pid, pci_bus_id:t.pci_bus_id, ...s}, null, 2);
}
trial.onchange = () => { fill(step, d.step_trials[Number(trial.value)].steps.map(s => s.step)); inspect(); };
step.onchange = row.oninput = inspect;
trial.value = Math.max(0, d.step_trials.findIndex(t => t.mode === 'ATLAS sharing'));
trial.onchange();
get('machine').textContent = d.machine.cpu + ' · ' + d.machine.logical_cpus + ' logical CPUs visible in ' + d.machine.environment;
const picker = get('machine-pick');
const sampleTime = s => new Date(s.utc_ns / 1e6).toISOString();
if (d.machine_samples.length) {
  fill(picker, d.machine_samples.map(s => sampleTime(s) + ' · ' + (s.trial || 'Between trials')));
  picker.onchange = () => {
    const s = d.machine_samples[Number(picker.value)];
    get('monitoring').textContent = 'Linux CPU busy: ' + fmt(s.linux_cpu_percent, '%') + ' · ATLAS worker CPU: ' + fmt(s.native_cpu_percent_one_core, '% of one core') + ' · Worker memory: ' + fmt(s.native_memory_bytes == null ? null : s.native_memory_bytes / 1048576, 'MiB');
    get('power').textContent = s.gpus.map(g => g.name + ' · Device busy: ' + fmt(g['utilization.gpu'], '%') + ' · Memory: ' + fmt(g['memory.used'], 'MiB') + ' · Temperature: ' + fmt(g['temperature.gpu'], '°C') + ' · Clock: ' + fmt(g['clocks.current.graphics'], 'MHz') + ' · Power: ' + fmt(g['power.draw'], 'W')).join(' | ') || s.gpu_note;
  };
  picker.onchange();
} else {
  get('sample-label').hidden = true;
  get('monitoring').textContent = d.monitoring_error || (d.monitoring ? 'No samples completed during this run.' : 'Machine monitoring was turned off for this comparison.');
}
if (d.monitoring_error && d.machine_samples.length) get('machine').append(' · ' + d.monitoring_error);
"""
    if not data['synthetic']:
        template = template.replace('Synthetic locations', 'Imported locations').replace(
            'Coordinates are synthetic east–west / north–south units.', 'Coordinates use the units provided in your file.'
        ).replace('Fixed seed 1729, four CPU workers', 'Imported CSV, four CPU workers')
    template = template.replace('let grouped=false;', """
const xValues=d.points.map(p=>p[0]), yValues=d.points.map(p=>p[1]);
const minX=Math.min(...xValues), minY=Math.min(...yValues);
const spanX=Math.max(...xValues)-minX||1, spanY=Math.max(...yValues)-minY||1;
let grouped=false;
""").replace('(x+10)/120', '(x-minX)/spanX').replace('(y+10)/120', '(y-minY)/spanY').replace(
        'East–west coordinate (synthetic units)', 'X coordinate'
    ).replace('32 rounds in every trial', f"{data['iterations']} rounds in every trial").replace(
        '32 grouping rounds, automatic clocks', f"{data['iterations']} grouping rounds, automatic clocks"
    ).replace('(r.round-1)/31', '(r.round-1)/Math.max(1,d.iterations-1)').replace(
        'Grouping round 1 → 32', f"Grouping step 1 to {data['iterations']}"
    ).replace('</script></html>', inspector + '</script></html>')
    return template.replace('__ROWS__', rows).replace('__FASTEST__', html.escape(fastest['mode'])).replace(
        '__DATE__', time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())).replace('__DATA__', payload)


def run(build, output, size, repeats=3, input_csv=None, monitor_machine=False, rounds=32):
    binary = (build / 'atlas_kmeans').resolve()
    if not binary.is_file():
        raise ValueError('Build atlas_kmeans in WSL first; see README.md.')
    if input_csv is None:
        rng = random.Random(1729)
        homes = [(18, 20), (78, 22), (25, 76), (79, 78)]
        points = [[rng.gauss(homes[i % 4][0], 7), rng.gauss(homes[i % 4][1], 7)] for i in range(size)]
        centers = [[x + 5, y - 5] for x, y in homes]
        ids = [str(i + 1) for i in range(size)]
        origin = {'kind': 'generated', 'seed': 1729}
    else:
        ids, points, centers, digest = import_points(input_csv)
        size = len(points)
        origin = {'kind': 'imported', 'name': input_csv.name, 'sha256': digest}
    output.mkdir(parents=True, exist_ok=False)
    progress = lambda message: write(output / 'progress.json', {'message': message})
    cancel_file = output / 'stop.request'
    dataset = output / 'locations.kmdat'
    kr.write_dataset(dataset, points, centers, origin)
    sample_count = min(size, 500)
    sample = [i * (size - 1) // max(1, sample_count - 1) for i in range(sample_count)]
    write(output / 'preview.json', {'points': [points[i] for i in sample], 'size': size})
    modes = [('CPU only', size, 'static'), ('GPU only', 0, 'static'), ('ATLAS sharing', size // 2, 'paper_proportional')]
    trials = []
    reference = None
    trace = None
    final = None
    samples = []
    steps = []

    def publish(sample):
        samples.append(sample)
        with (output / 'machine.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(sample, allow_nan=False) + '\n')
        write(output / 'machine-live.json', sample)

    monitor = Monitor(publish) if monitor_machine else None
    try:
        if monitor:
            monitor.start()
        for block in range(repeats):
            order = list(modes)
            random.Random(8100 + block).shuffle(order)
            for name, cpu, policy in order:
                if cancel_file.exists():
                    raise RuntimeError('Stopped by user')
                run_id = f'trial-{len(trials):02d}'
                progress(f'Round {block + 1}/{repeats}: {name} | {size:,} locations, {rounds} grouping rounds')
                live = {'phase': 'starting', 'mode': name, 'block': block + 1, 'run_id': run_id,
                        'size': size, 'points': [points[i] for i in sample], 'trace': [],
                        'centroids': centers, 'completed': list(trials), 'labels': None,
                        'input_kind': origin['kind'], 'rounds': rounds}
                write(output / 'live.json', live)
                last_update = 0
                def observe(row):
                    nonlocal last_update
                    live['phase'] = 'computing'
                    live['round'] = row['index'] + 1
                    live['assignment'] = step_evidence(row, {}, size)
                    live['trace'].append({'round': row['index'] + 1, 'cpu_percent': 100 * row['cpu_items'] / size})
                    values = row['centroids']
                    live['centroids'] = list(zip(values[::2], values[1::2]))
                    now = time.monotonic()
                    if now - last_update >= .1 or row['index'] == rounds - 1:
                        write(output / 'live.json', live)
                        last_update = now
                cfg = Config(run_id, size, cpu, mode='allocation_only', allocation_policy=policy,
                             allocation_iterations=4, allocation_interval_ns=0, cadence_profile='batched_epochs_v1')
                case = dict(workload='kmeans', integration=asdict(cfg), steps=rounds, workers=4,
                            native_mode='fixed', validation_mode='timing', reduction='hierarchical',
                            transfer_mode='streamed', feedback_batch=4, telemetry=False, runtime_version='runtime-2.0')
                pid = None

                def started(native_pid):
                    nonlocal pid
                    pid = native_pid
                    if monitor:
                        monitor.target = (run_id, native_pid)

                try:
                    result = execute(binary, dataset, None, case, None, timeout=90, on_iteration=observe,
                                     on_process=started, cancelled=cancel_file.exists)
                finally:
                    if monitor:
                        monitor.target = (None, None)
                result['linux_pid'] = pid
                write(output / (run_id + '.json'), {'case': case, 'record': result})
                if result['status'] != 'ok':
                    raise RuntimeError(f'{name}: {result["stderr"].strip() or result["failure"]}. Raw failure retained in {run_id}.json.')
                replay_decisions(case, result)
                rows = [json.loads(line) for line in result['stdout'].splitlines() if line.startswith('{')]
                summary = rows[-1]
                if summary.get('kind') != 'summary' or summary.get('status') != 'passed' or summary.get('independent_validation') != 'passed' or summary.get('iterations') != rounds:
                    raise ValueError(f'{name}: missing independent correctness validation')
                labels, centroids = summary['final_labels'], summary['final_centroids']
                if len(labels) != size or len(centroids) != 8 or any(type(label) is not int or not 0 <= label < 4 for label in labels):
                    raise ValueError('Incomplete grouping output')
                if reference is None:
                    reference = (labels, centroids)
                elif labels != reference[0] or not all(kr.close(a, b) for a, b in zip(centroids, reference[1])):
                    raise ValueError('CPU/GPU grouping outputs differ; no comparison published.')
                iterations = [row for row in rows if row.get('kind') == 'iteration' and not row['warmup']]
                barriers = result.get('barriers', [])
                configuration = next((row for row in rows if row.get('kind') == 'configuration'), {})
                steps.append({'trial': run_id, 'mode': name, 'linux_pid': pid,
                              'pci_bus_id': configuration.get('pci_bus_id'),
                              'steps': [step_evidence(row, barriers[i] if i < len(barriers) else {}, size)
                                        for i, row in enumerate(iterations)]})
                trials.append({'mode': name, 'block': block, 'record': run_id + '.json',
                               'workload_ms': summary['loop_elapsed_ns'] / 1e6,
                               'command_ms': result['elapsed_ns'] / 1e6})
                live.update(phase='validated', labels=[labels[i] for i in sample], completed=list(trials))
                write(output / 'live.json', live)
                if name == 'ATLAS sharing' and trace is None:
                    trace = [{'round': row['index'] + 1, 'cpu_percent': 100 * row['cpu_items'] / size} for row in iterations]
                    final = {'labels': [labels[i] for i in sample], 'centroids': list(zip(centroids[::2], centroids[1::2])),
                             'counts': [labels.count(i) for i in range(4)]}
        if monitor:
            monitor.close()
        if cancel_file.exists():
            raise RuntimeError('Stopped by user')
        with (output / 'groups.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['row', 'id', 'x', 'y', 'group'])
            for i, (identity, point, label) in enumerate(zip(ids, points, reference[0]), 1):
                # Spreadsheet applications interpret formula-prefixed IDs as expressions.
                safe_id = "'" + identity if identity.startswith(('=', '+', '-', '@')) else identity
                writer.writerow([i, safe_id, *point, label + 1])
        summaries = []
        for name, _, _ in modes:
            rows = [trial for trial in trials if trial['mode'] == name]
            summaries.append({'mode': name, 'workload_ms': statistics.median(r['workload_ms'] for r in rows),
                              'min_ms': min(r['workload_ms'] for r in rows), 'max_ms': max(r['workload_ms'] for r in rows),
                              'command_ms': statistics.median(r['command_ms'] for r in rows)})
        result = {'status': 'passed', 'size': size, 'repeats': repeats, 'iterations': rounds,
                  'synthetic': input_csv is None, 'seed': origin.get('seed'), 'points': [points[i] for i in sample], **final,
                  'evidence_version': 1, 'input': origin, 'step_trials': steps,
                  'machine_samples': samples, 'monitoring': monitor_machine,
                  'monitoring_error': monitor.error if monitor else None,
                  'machine': {'system': platform.release(), 'cpu': cpu_name(),
                              'logical_cpus': os.cpu_count(), 'environment': 'Linux guest' if 'microsoft' in platform.release().lower() else 'Linux'},
                  'modes': summaries, 'trace': trace, 'trials': trials,
                  'allocation_changes': sum(a['cpu_percent'] != b['cpu_percent'] for a, b in zip(trace, trace[1:])),
                  'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
                  'source_sha256': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in [ROOT / 'scripts/demo_workload.py', *sorted((ROOT / 'src/greengpu').glob('*.py'))]},
                  'settings': {'workers': 4, 'frequency': 'unchanged automatic clocks', 'python': sys.version},
                  'correctness': 'Native independent reference passed in every trial; final labels and centroids match across modes; controller decisions replayed.',
                  'scope': 'Illustrative single-host comparison, automatic clocks, 4 CPU workers, including live-observation overhead. Not a statistical performance qualification or delivery-route optimizer.',
                  'energy_savings': None}
        write(output / 'summary.json', result)
        live.update(phase='complete', completed=list(trials))
        write(output / 'live.json', live)
        (output / 'demo.html').write_text(presentation(result), encoding='utf-8')
        inventory = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file() and p.name not in ('progress.json', 'live.json', 'checksums.json', 'machine-live.json', 'stop.request')}
        write(output / 'checksums.json', inventory)
        progress('Completed. All grouping results match; comparison ready.')
        return result
    except Exception as error:
        if monitor and monitor.thread.is_alive():
            monitor.close()
        write(output / 'failure.json', {'error': str(error)})
        write(output / 'live.json', {'phase': 'failed', 'error': str(error)})
        progress('Run failed. Partial evidence retained; no successful comparison published.')
        raise


def cpu_name():
    try:
        with Path('/proc/cpuinfo').open() as stream:
            return next((line.split(':', 1)[1].strip() for line in stream if line.startswith('model name')), platform.machine())
    except OSError:
        return platform.machine()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size', type=int, choices=(2000, 16000, 64000), default=16000)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument('--input', type=Path, help='CSV with id,x,y columns')
    inputs.add_argument('--windows-input', help='Windows path to a CSV')
    parser.add_argument('--monitor', action='store_true', help='Observe Linux CPU and available whole-GPU measurements')
    parser.add_argument('--steps', type=int, choices=(32, 512), default=32, help='Grouping steps per trial')
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('Run the workload inside Linux/WSL.')
    started = time.monotonic()
    try:
        path = args.input
        if args.windows_input:
            path = Path(subprocess.check_output(['wslpath', '-a', '-u', args.windows_input], text=True, timeout=5).rstrip('\r\n'))
        result = run(args.build, args.output, args.size, input_csv=path, monitor_machine=args.monitor, rounds=args.steps)
        print(json.dumps({'status': result['status'], 'modes': result['modes'], 'elapsed_s': time.monotonic() - started}))
        return 0
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
