#!/usr/bin/env python3
"""Small, bounded presentation workload using the existing real native runner."""
import argparse
from dataclasses import asdict
import hashlib
import html
import json
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
<footer>This is a self-contained recorded demonstration. It can be presented offline without WSL or a GPU. The launcher runs fresh computations when requested; this page never claims to be live.</footer></main>
<script>const d=__DATA__,ns='http://www.w3.org/2000/svg',colors=['#377bb5','#c57427','#328460','#8662a8'];function el(tag,attrs,text){let e=document.createElementNS(ns,tag);Object.entries(attrs||{}).forEach(([k,v])=>e.setAttribute(k,v));if(text!==undefined)e.textContent=text;return e}function text(svg,x,y,s,extra={}){svg.append(el('text',{x,y,...extra},s))}document.querySelector('#size').textContent=d.size.toLocaleString();document.querySelector('#sample').textContent=`Showing ${d.points.length} sampled stops; all ${d.size.toLocaleString()} stops were processed.`;document.querySelector('#changes').textContent=d.allocation_changes;
let grouped=false;function map(){const svg=document.querySelector('#map');svg.replaceChildren();text(svg,250,325,'East–west coordinate (synthetic units)',{'text-anchor':'middle'});for(let i=0;i<d.points.length;i++){let [x,y]=d.points[i];svg.append(el('circle',{cx:25+(x+10)/120*450,cy:295-(y+10)/120*270,r:3,fill:grouped?colors[d.labels[i]]:'#92a1a6',opacity:.8}))}document.querySelector('#legend').replaceChildren();if(grouped){d.counts.forEach((n,i)=>{let span=document.createElement('span'),dot=document.createElement('i');dot.className='dot';dot.style.background=colors[i];span.append(dot,`Area ${i+1}: ${n.toLocaleString()} stops`);document.querySelector('#legend').append(span)})}document.querySelector('#toggle').textContent=grouped?'Show original locations':'Show the groups'}document.querySelector('#toggle').onclick=()=>{grouped=!grouped;map()};map();
const bars=document.querySelector('#times'),max=Math.max(...d.modes.map(m=>m.workload_ms))*1.12;d.modes.forEach((m,i)=>{let y=30+i*75,w=m.workload_ms/max*300;text(bars,5,y+20,m.mode);bars.append(el('rect',{x:145,y,width:w,height:32,fill:i===2?'#176c55':'#8b9fa7'}));text(bars,150+w,y+21,m.workload_ms.toFixed(2)+' ms')});text(bars,250,278,'Median workload time (milliseconds) · lower is better',{'text-anchor':'middle'});
const tr=document.querySelector('#trace');for(let p of [0,50,100]){let y=185-p*1.5;tr.append(el('line',{x1:42,y1:y,x2:478,y2:y,stroke:'#e1e7e9'}));text(tr,35,y+4,p+'%',{'text-anchor':'end'})}tr.append(el('polyline',{points:d.trace.map(r=>`${42+(r.round-1)/31*436},${185-r.cpu_percent*1.5}`).join(' '),fill:'none',stroke:'#176c55','stroke-width':3}));text(tr,250,221,'Grouping round 1 → 32 (first ATLAS trial)',{'text-anchor':'middle'});text(tr,45,20,'CPU share (%) · GPU receives the rest');</script></html>'''
    return template.replace('__DATA__', payload).replace('__ROWS__', rows).replace('__FASTEST__', html.escape(fastest['mode'])).replace('__DATE__', time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime()))


def run(build, output, size, repeats=3):
    binary = (build / 'atlas_kmeans').resolve()
    if not binary.is_file():
        raise ValueError('Build atlas_kmeans in WSL first; see README.md.')
    output.mkdir(parents=True, exist_ok=False)
    progress = lambda message: write(output / 'progress.json', {'message': message})
    rng = random.Random(1729)
    homes = [(18, 20), (78, 22), (25, 76), (79, 78)]
    points = [[rng.gauss(homes[i % 4][0], 7), rng.gauss(homes[i % 4][1], 7)] for i in range(size)]
    centers = [[x + 5, y - 5] for x, y in homes]
    dataset = output / 'locations.kmdat'
    kr.write_dataset(dataset, points, centers, {'kind': 'synthetic_delivery_locations', 'seed': 1729})
    sample = list(range(0, size, max(1, size // 500)))[:500]
    write(output / 'preview.json', {'points': [points[i] for i in sample], 'size': size})
    modes = [('CPU only', size, 'static'), ('GPU only', 0, 'static'), ('ATLAS sharing', size // 2, 'paper_proportional')]
    trials = []
    reference = None
    trace = None
    final = None
    try:
        for block in range(repeats):
            order = list(modes)
            random.Random(8100 + block).shuffle(order)
            for name, cpu, policy in order:
                run_id = f'trial-{len(trials):02d}'
                progress(f'Round {block + 1}/{repeats}: {name} | {size:,} locations, 32 grouping rounds')
                live = {'phase': 'starting', 'mode': name, 'block': block + 1, 'run_id': run_id,
                        'size': size, 'points': [points[i] for i in sample], 'trace': [],
                        'centroids': centers, 'completed': list(trials), 'labels': None}
                write(output / 'live.json', live)
                last_update = 0
                def observe(row):
                    nonlocal last_update
                    live['phase'] = 'computing'
                    live['round'] = row['index'] + 1
                    live['trace'].append({'round': row['index'] + 1, 'cpu_percent': 100 * row['cpu_items'] / size})
                    values = row['centroids']
                    live['centroids'] = list(zip(values[::2], values[1::2]))
                    now = time.monotonic()
                    if now - last_update >= .1 or row['index'] == 31:
                        write(output / 'live.json', live)
                        last_update = now
                cfg = Config(run_id, size, cpu, mode='allocation_only', allocation_policy=policy,
                             allocation_iterations=4, allocation_interval_ns=0, cadence_profile='batched_epochs_v1')
                case = dict(workload='kmeans', integration=asdict(cfg), steps=32, workers=4,
                            native_mode='fixed', validation_mode='timing', reduction='hierarchical',
                            transfer_mode='streamed', feedback_batch=4, telemetry=False, runtime_version='runtime-2.0')
                result = execute(binary, dataset, None, case, None, timeout=90, on_iteration=observe)
                write(output / (run_id + '.json'), {'case': case, 'record': result})
                if result['status'] != 'ok':
                    raise RuntimeError(f'{name}: {result["stderr"].strip() or result["failure"]}. Raw failure retained in {run_id}.json.')
                replay_decisions(case, result)
                rows = [json.loads(line) for line in result['stdout'].splitlines() if line.startswith('{')]
                summary = rows[-1]
                if summary.get('kind') != 'summary' or summary.get('status') != 'passed' or summary.get('independent_validation') != 'passed' or summary.get('iterations') != 32:
                    raise ValueError(f'{name}: missing independent correctness validation')
                labels, centroids = summary['final_labels'], summary['final_centroids']
                if len(labels) != size or len(centroids) != 8 or any(type(label) is not int or not 0 <= label < 4 for label in labels):
                    raise ValueError('Incomplete grouping output')
                if reference is None:
                    reference = (labels, centroids)
                elif labels != reference[0] or not all(kr.close(a, b) for a, b in zip(centroids, reference[1])):
                    raise ValueError('CPU/GPU grouping outputs differ; no comparison published.')
                iterations = [row for row in rows if row.get('kind') == 'iteration' and not row['warmup']]
                trials.append({'mode': name, 'block': block, 'record': run_id + '.json',
                               'workload_ms': summary['loop_elapsed_ns'] / 1e6,
                               'command_ms': result['elapsed_ns'] / 1e6})
                live.update(phase='validated', labels=[labels[i] for i in sample], completed=list(trials))
                write(output / 'live.json', live)
                if name == 'ATLAS sharing' and trace is None:
                    trace = [{'round': row['index'] + 1, 'cpu_percent': 100 * row['cpu_items'] / size} for row in iterations]
                    final = {'labels': [labels[i] for i in sample], 'centroids': list(zip(centroids[::2], centroids[1::2])),
                             'counts': [labels.count(i) for i in range(4)]}
        summaries = []
        for name, _, _ in modes:
            rows = [trial for trial in trials if trial['mode'] == name]
            summaries.append({'mode': name, 'workload_ms': statistics.median(r['workload_ms'] for r in rows),
                              'min_ms': min(r['workload_ms'] for r in rows), 'max_ms': max(r['workload_ms'] for r in rows),
                              'command_ms': statistics.median(r['command_ms'] for r in rows)})
        result = {'status': 'passed', 'size': size, 'repeats': repeats, 'iterations': 32,
                  'synthetic': True, 'seed': 1729, 'points': [points[i] for i in sample], **final,
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
        inventory = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file() and p.name not in ('progress.json', 'live.json', 'checksums.json')}
        write(output / 'checksums.json', inventory)
        progress('Completed. All grouping results match; comparison ready.')
        return result
    except Exception as error:
        write(output / 'failure.json', {'error': str(error)})
        write(output / 'live.json', {'phase': 'failed', 'error': str(error)})
        progress('Run failed. Partial evidence retained; no successful comparison published.')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size', type=int, choices=(2000, 16000, 64000), default=16000)
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('Run the workload inside Linux/WSL.')
    started = time.monotonic()
    try:
        result = run(args.build, args.output, args.size)
        print(json.dumps({'status': result['status'], 'modes': result['modes'], 'elapsed_s': time.monotonic() - started}))
        return 0
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
