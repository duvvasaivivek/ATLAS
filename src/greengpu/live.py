"""Guarded barrier-based joint control, separate from immutable shadow contracts.

A platform provider owns independently qualified telemetry and policy readback.
There is deliberately no JSON/CLI switch that declares this machine qualified.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

from . import frequency as f
from .actuation import ActuationBlocked, ClockTransaction, valid_policy
from .energy import finite_nonnegative, integrate_power
from .experiment import save_new
from .frequency_experiment import sha
from .frequency_hardware import verify_setting
from .integration import Coordinator

SCHEMA = 'live-joint-1.0'
ENERGY_DOMAINS = ('gpu_board', 'cpu_package', 'system_wall')


class DefaultProvider:
    """Read-only default. Missing calibration never turns into live permission."""
    def __init__(self):
        from .actuation import NvmlClockBackend
        self.backend=NvmlClockBackend()
    def qualification(self):
        cap=self.backend.discover()
        return dict(device_uuid=cap['device_uuid'],pci_bus_id=cap['pci_bus_id'],
                    evidence_kind='hardware',telemetry_clock='blocked',restoration_probe='blocked',
                    device_timing='blocked',verified_pairs=[],evidence_id=None)
    def samples(self):raise ActuationBlocked('No qualified default telemetry adapter')
    def close(self):self.backend.close()


def load_provider(name=None):
    if name is None:return DefaultProvider()
    import importlib
    module,separator,attribute=name.partition(':')
    if not separator or not all(p.isidentifier() for p in module.split('.')) or not attribute.isidentifier():
        raise ValueError('Provider must name an installed Python module:factory')
    provider=getattr(importlib.import_module(module),attribute)()
    if not all(callable(getattr(provider,k,None)) for k in ('qualification','samples','close')):
        raise TypeError('Provider must implement qualification(), samples(), close() and backend')
    return provider


def preflight(provider, config, now_ns):
    """Provider attestations are bound to a fresh backend snapshot and GPU identity."""
    q = deepcopy(provider.qualification())
    snapshot = provider.backend.snapshot()
    required = ('telemetry_clock', 'restoration_probe', 'device_timing')
    if (not valid_policy(snapshot) or q.get('device_uuid') != snapshot['device_uuid'] or
            q.get('device_uuid') != provider.backend.device_uuid or
            q.get('evidence_kind') not in ('hardware', 'software_test') or
            not q.get('pci_bus_id') or any(q.get(k) != 'qualified' for k in required) or
            not q.get('evidence_id')):
        raise ActuationBlocked('Live control requires device-bound qualification, restorable policy and qualified timing')
    if (type(q.get('qualified_at_ns')) is not int or type(q.get('valid_until_ns')) is not int or
            not 0 <= q['qualified_at_ns'] <= now_ns < q['valid_until_ns']):
        raise ActuationBlocked('Live qualification is stale or has an invalid time boundary')
    pairs = {tuple(p) for p in q.get('verified_pairs', ())}
    if not config.pairs or not set(config.pairs) <= pairs:
        raise ActuationBlocked('Every requested frequency pair must be verified under load')
    if config.initial_cpu_items == config.n:
        raise ValueError('Initial live verification requires nonempty GPU work')
    if config.mode != 'allocation_shadow':
        raise ValueError('Live orchestration requires an explicit allocation_shadow policy configuration')
    from .actuation import NvmlClockBackend
    if q['evidence_kind'] == 'hardware' and type(provider.backend) is not NvmlClockBackend:
        raise ActuationBlocked('Hardware evidence requires the physical NVML backend')
    if q['evidence_kind'] == 'software_test':
        if isinstance(provider.backend, NvmlClockBackend):
            raise ActuationBlocked('Test qualification cannot authorize the physical NVML actuator')
    return q


class LiveController:
    """Deterministic control transitions; setters remain in the transaction owner."""
    def __init__(self, config, qualification, settling_timeout_ns=10_000_000_000, *, frequency_control=True):
        if type(settling_timeout_ns) is not int or settling_timeout_ns <= 0:
            raise ValueError('Positive settling deadline required')
        self.config = config
        self.q = qualification
        self.engine = Coordinator(replace(config, mode='allocation_only'))
        self.frequency = f.Config(config.pairs, policy=config.frequency_policy,
                                  provenance='hardware_reported_unverified')
        self.fstate = f.initial_state(self.frequency, config.run_id)
        self.frames = []
        self.last_stamp = -1
        self.current = None
        self.request_end = None
        self.pending = True
        self.last_verified_ns = None
        self.last_frequency_ns = 0
        self.timeout_ns = settling_timeout_ns
        self.frequency_control = frequency_control

    def applied(self, pair, end_ns):
        if tuple(pair) not in self.config.pairs or type(end_ns) is not int or end_ns < 0:
            raise ValueError('Invalid acknowledged setting')
        self.current = tuple(pair)
        self.request_end = end_ns
        self.pending = True
        self.frames = []
        self.engine.window = []
        self.last_frequency_ns = end_ns

    def barrier(self, observation, now_ns, new_frames=(), final=False, phase=0):
        if self.request_end is None or now_ns < self.request_end:
            raise ValueError('No acknowledged setting before live work')
        if now_ns >= self.q['valid_until_ns']:
            raise ActuationBlocked('Live qualification expired')
        for frame in new_frames:
            if frame.get('device_uuid') != self.q['device_uuid']:
                raise ActuationBlocked('Cross-device live telemetry')
            start, end = frame.get('acquisition_start_ns'), frame.get('acquisition_end_ns')
            stamp = frame.get('sample_timestamp_us')
            mapped = frame.get('sample_monotonic_estimate_ns')
            if (type(start) is not int or type(end) is not int or type(mapped) is not int or
                    not 0 <= mapped <= end or not 0 <= start <= end <= now_ns or
                    type(stamp) is not int or stamp <= self.last_stamp):
                raise ActuationBlocked('Invalid, future or repeated live telemetry')
            if not f.finite(frame.get('temperature_c')) or not 0 <= frame['temperature_c'] < 85:
                raise ActuationBlocked('Invalid temperature or temperature limit reached')
            self.last_stamp = stamp
            self.frames.append(deepcopy(frame))
        # A verification streak must also be fresh at this barrier, not just at acquisition.
        self.frames = [r for r in self.frames if now_ns-r['sample_monotonic_estimate_ns'] <= 500_000_000]
        if len(self.frames) > 10000:
            raise ActuationBlocked('Unbounded telemetry rate')
        verification = verify_setting(self.frames, self.current, self.request_end, require_latest=True)
        verified = verification['status'] == 'verified_under_load'
        if verified:
            self.last_verified_ns = now_ns
        elif (not self.pending and self.last_verified_ns is not None and
              now_ns-self.last_verified_ns > self.timeout_ns):
            raise ActuationBlocked('Loaded clock verification was lost beyond the allowed deadline')
        was_pending = self.pending
        if self.pending and verified:
            self.pending = False
            self.engine.capacity_epoch += 1
            self.engine.window = []
            self.engine.since_ns = now_ns
            self.engine.last_allocation_ns = now_ns
            self.engine.last_frequency_ns = now_ns
            self.engine.settling = self.config.settle_iterations
        elif self.pending and now_ns-self.request_end > self.timeout_ns:
            raise ActuationBlocked('Clock setting did not verify before the settling deadline')
        old_since = self.engine.since_ns
        old_epoch = self.engine.partition_epoch+self.engine.phase
        allocation = self.engine.barrier(observation, now_ns, final=final, phase=phase,
                                         action_boundary=verified and not was_pending)
        if not verified or was_pending:
            self.engine.window = []  # Never learn throughput across unverified capacity.
        result = dict(allocation=allocation, verification=verification,
                      next_cpu_items=allocation['next_cpu_items'], frequency=None,
                      requested_pair=None, setting_pending=self.pending,
                      reason='settling' if self.pending else 'hold_unverified_clock' if not verified else 'collect')
        if final:
            result['reason'] = ('incomplete_settling' if self.pending else
                                'workload_complete' if verified else 'incomplete_verification')
            return result
        if (self.frequency_control and verified and not was_pending and old_since is not None and
                now_ns-self.last_frequency_ns >= self.config.frequency_interval_ns):
            sample = self.frames[-1]
            if sample['sample_monotonic_estimate_ns'] >= old_since+self.config.frequency_guard_ns:
                window = dict(kind='nvml_buffer_timestamp_us', start=None, end=sample['sample_timestamp_us'])
                obs = dict(run_id=self.config.run_id, sample_id=observation['iteration'], epoch=old_epoch,
                           core=sample['core_utilization'], memory=sample['memory_utilization'],
                           core_window=window, memory_window=dict(window), valid=True,
                           semantics='nvml_kernel_and_memory_busy_time_fraction',
                           source_ref=f"qualified:{self.q['evidence_id']}:{sample['sample_timestamp_us']}")
                decision = f.transition(self.frequency, self.fstate, obs)
                self.fstate = decision['post_state']
                self.last_frequency_ns = now_ns
                result['frequency'] = decision
                pair = tuple(decision['recommended_pair']) if decision['recommended_pair'] else None
                if decision['reason'] == 'recommend_pair' and pair != self.current:
                    result['requested_pair'] = list(pair)
        return result


def meter_spec(qualification):
    return dict(domain=qualification.get('energy_domain'), evidence_id=qualification.get('energy_evidence_id'),
                max_gap_ns=qualification.get('energy_max_gap_ns'), accuracy_w=qualification.get('energy_accuracy_w'),
                accuracy_relative=qualification.get('energy_accuracy_relative'))


def valid_meter(spec):
    return (spec['domain'] in ENERGY_DOMAINS and isinstance(spec['evidence_id'], str) and bool(spec['evidence_id'].strip()) and
            type(spec['max_gap_ns']) is int and spec['max_gap_ns'] > 0 and
            finite_nonnegative(spec['accuracy_w']) and f.fraction(spec['accuracy_relative']) and spec['accuracy_relative'] < 1)


def energy_contract(provider, qualification):
    spec = meter_spec(qualification)
    if (not callable(getattr(provider, 'energy_samples', None)) or not valid_meter(spec) or
            spec['evidence_id'] == qualification.get('evidence_id')):
        raise ActuationBlocked('Energy requires an independent calibrated power sampler, domain, gap limit and accuracy bounds')
    return spec


def measure_energy(contract, frames, start, end):
    if not valid_meter(contract):
        raise ValueError('Invalid energy measurement contract')
    if any(frame.get('domain') != contract['domain'] or frame.get('evidence_id') != contract['evidence_id']
           for frame in frames):
        raise ValueError('Power samples differ from the calibrated energy domain or source')
    value = integrate_power(frames, start, end, contract['max_gap_ns'])
    uncertainty_j = None
    if value['status'] == 'ok':
        uncertainty_j = (contract['accuracy_w']*(end-start)/1e9 + contract['accuracy_relative']*value['joules'])
        if not finite_nonnegative(uncertainty_j):
            raise ValueError('Nonfinite energy uncertainty')
    return dict(value, domain=contract['domain'], evidence_id=contract['evidence_id'], meter_accuracy_j=uncertainty_j)


def run_live(provider, workload, config, output, clock=time.monotonic_ns, *, initial_pair=None,
             frequency_control=True, measure=False):
    """Execute a qualified provider and barrier workload, restoring on every exit.

    Workload interface: start(config, qualification), release(cpu_items),
    observe(run_id)->(observation, final, phase), finish()->correctness record,
    close(). start must verify the CUDA device and leave execution armed/paused.
    Provider supplies qualification(), backend, samples(); all platform I/O must
    have a bounded implementation. No actuation proceeds from a user assertion.
    """
    qualification = preflight(provider, config, clock())
    if type(frequency_control) is not bool or type(measure) is not bool:
        raise ValueError('Expected boolean frequency control and measurement options')
    pair = config.pairs[0] if initial_pair is None else tuple(initial_pair)
    if pair not in config.pairs:
        raise ActuationBlocked('Initial pair must be among the qualified settings')
    measurement = energy_contract(provider, qualification) if measure else None
    root = Path(output)
    from .live_native import NativeWorkload
    if isinstance(workload, NativeWorkload) and workload.directory.resolve() != (root/'native').resolve():
        raise ValueError('Native evidence must be retained under the live archive')
    if qualification['evidence_kind']=='hardware' and not isinstance(workload, NativeWorkload):
        raise ActuationBlocked('Hardware claims require the native workload verifier')
    root.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema_version=SCHEMA, config=asdict(config), qualification=qualification)
    if not frequency_control or initial_pair is not None or measurement is not None:
        manifest.update(initial_pair=list(pair), frequency_control=frequency_control, measurement=measurement)
    save_new(root/'configuration.json', manifest)
    controller = LiveController(config, qualification, frequency_control=frequency_control)
    records = []
    result = None
    tx = None
    started = clock()
    try:
        with ClockTransaction(provider.backend, root/'clocks', config.pairs) as tx:
            try:
                workload.start(config, qualification)
                if clock() >= qualification['valid_until_ns']:
                    raise ActuationBlocked('Live qualification expired before clock actuation')
                end = tx.apply(pair)
                controller.applied(pair, end)
                tx.record('live_initial_setting', pair=list(pair), request_end_ns=end)
                release_ns = clock()
                if measurement is not None:
                    tx.record('live_measurement_start', durable=False, boundary_ns=release_ns)
                workload.release(config.initial_cpu_items)
                while True:
                    obs, final, phase = workload.observe(config.run_id)
                    frames = provider.samples()
                    now = clock()
                    decision = controller.barrier(obs, now, frames, final, phase)
                    entry = dict(observation=obs, now_ns=now, frames=frames, final=final, phase=phase,
                                 decision=decision, applied_end_ns=None)
                    records.append(entry)
                    # The next durable clock request or restoration flushes observations.
                    tx.record('live_barrier', durable=False, entry=dict(entry))
                    if final:
                        result = workload.finish()
                        if result.get('status') != 'passed':
                            raise RuntimeError('Final independent workload validation failed')
                        if measurement is not None:
                            completion_ns = clock()
                            tx.record('live_measurement_end', durable=False, boundary_ns=completion_ns)
                        break
                    if decision['requested_pair'] is not None:
                        if clock() >= qualification['valid_until_ns']:
                            raise ActuationBlocked('Live qualification expired before clock actuation')
                        end = tx.apply(decision['requested_pair'])
                        controller.applied(decision['requested_pair'], end)
                        entry['applied_end_ns'] = end
                        tx.record('live_applied', iteration=obs['iteration'], pair=decision['requested_pair'], request_end_ns=end)
                    workload.release(decision['next_cpu_items'])
            finally:
                # Stop native work before restoration, including every failure path.
                workload.close()
        status = 'passed' if decision['reason']=='workload_complete' else decision['reason']
        result = dict(schema_version=SCHEMA, status=status, evidence_kind=qualification['evidence_kind'],
                      physical_control_evidence=qualification['evidence_kind']=='hardware' and status=='passed',
                      correctness=result, records=records, initial_request_end_ns=next(e['request_end_ns'] for e in tx.events if e['kind']=='live_initial_setting'),
                      restoration='verified', elapsed_ns=clock()-started, energy_savings=None)
        if measurement is not None:
            # The calibrated sampler starts before the run and brackets the full released workload.
            frames = list(provider.energy_samples(release_ns, completion_ns))
            value = measure_energy(measurement, frames, release_ns, completion_ns)
            save_new(root/'power.json', dict(samples=frames, start_ns=release_ns, end_ns=completion_ns))
            result['energy'] = dict(value, physical_measurement=qualification['evidence_kind']=='hardware' and value['status']=='ok')
        save_new(root/'result.json', result)
    except BaseException as error:
        save_new(root/'failure.json', dict(error=f'{type(error).__name__}: {error}', records=records,
                                         restoration_events=[e for e in (tx.events if tx else []) if e['kind']=='restoration']))
        raise
    finally:
        save_new(root/'checksums.json', {p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*')) if p.is_file()})
    return result


def replay(root):
    from .integration import Config
    root=Path(root)
    hashes=json.loads((root/'checksums.json').read_text())
    files={p.relative_to(root).as_posix():p for p in root.rglob('*') if p.is_file() and p.name!='checksums.json'}
    if files.keys()!=hashes.keys() or any(p.is_symlink() or sha(p)!=hashes[n] for n,p in files.items()):
        raise ValueError('Live archive integrity failure')
    manifest=json.loads((root/'configuration.json').read_text())
    result=json.loads((root/'result.json').read_text())
    cfg=Config(**manifest['config'])
    frequency_control=manifest.get('frequency_control',True)
    if type(frequency_control) is not bool:
        raise ValueError('Invalid live frequency control setting')
    controller=LiveController(cfg,manifest['qualification'],frequency_control=frequency_control)
    pair=tuple(manifest.get('initial_pair',cfg.pairs[0]))
    controller.applied(pair,result['initial_request_end_ns'])
    if not result['records'] or any(r['final'] != (i==len(result['records'])-1) for i,r in enumerate(result['records'])):
        raise ValueError('Invalid live final boundary')
    for row in result['records']:
        expected=controller.barrier(row['observation'],row['now_ns'],row['frames'],row['final'],row['phase'])
        if expected!=row['decision']:raise ValueError('Live decision replay mismatch')
        if expected['requested_pair']:
            if type(row['applied_end_ns']) is not int or row['applied_end_ns']<row['now_ns']:raise ValueError('Missing or premature live acknowledgement')
            controller.applied(expected['requested_pair'],row['applied_end_ns'])
        elif row['applied_end_ns'] is not None:raise ValueError('Unexpected live acknowledgement')
    events=[json.loads(line) for line in (root/'clocks/journal.jsonl').read_text().splitlines()]
    if not events or events[-1].get('kind')!='restoration' or events[-1].get('status')!='verified':
        raise ValueError('Missing verified restoration')
    prior=events[0]['snapshot'];after=events[-1]['snapshot']
    if not valid_policy(after) or after['device_uuid']!=prior['device_uuid'] or after['policy']!=prior['policy']:
        raise ValueError('Restoration readback mismatch')
    entries=[e['entry'] for e in events if e['kind']=='live_barrier']
    if entries!=[dict(r,applied_end_ns=None) for r in result['records']]:raise ValueError('Live journal/barrier mismatch')
    requests=[e['pair'] for e in events if e['kind']=='request']
    if requests!=[list(pair)]+[r['decision']['requested_pair'] for r in result['records'] if r['decision']['requested_pair']]:
        raise ValueError('Live journal/request mismatch')
    initial=[e for e in events if e['kind']=='live_initial_setting']
    applied=[e for e in events if e['kind']=='live_applied']
    if len(initial)!=1 or initial[0]['request_end_ns']!=result['initial_request_end_ns'] or initial[0]['pair']!=list(pair):
        raise ValueError('Initial live acknowledgement mismatch')
    if [(e['iteration'],e['pair'],e['request_end_ns']) for e in applied]!=[(r['observation']['iteration'],r['decision']['requested_pair'],r['applied_end_ns']) for r in result['records'] if r['applied_end_ns'] is not None]:
        raise ValueError('Live acknowledgement journal mismatch')
    expected_status='passed' if expected['reason']=='workload_complete' else expected['reason']
    if result['status']!=expected_status or result['restoration']!='verified' or result['evidence_kind']!=manifest['qualification']['evidence_kind']:
        raise ValueError('Live summary mismatch')
    if result['physical_control_evidence']!=(result['evidence_kind']=='hardware' and expected_status=='passed'):
        raise ValueError('Invalid physical-control evidence claim')
    measurement=manifest.get('measurement')
    if measurement is not None:
        power=json.loads((root/'power.json').read_text())
        qualification=manifest['qualification']
        expected_meter=meter_spec(qualification)
        boundaries=[(e['kind'],e['boundary_ns'],e['monotonic_ns']) for e in events
                    if e['kind'] in ('live_measurement_start','live_measurement_end')]
        if (not result['initial_request_end_ns'] <= power['start_ns'] <= result['records'][0]['observation']['release_ns'] or
                not result['records'][-1]['now_ns'] <= power['end_ns'] <= events[-1]['monotonic_ns'] or
                [(kind, ns) for kind,ns,_ in boundaries] !=
                [('live_measurement_start',power['start_ns']),('live_measurement_end',power['end_ns'])] or
                any(ns > recorded for _,ns,recorded in boundaries) or measurement != expected_meter):
            raise ValueError('Energy window or calibration is not bound to the live trial')
        value=measure_energy(measurement,power['samples'],power['start_ns'],power['end_ns'])
        if result.get('energy')!=dict(value,physical_measurement=result['evidence_kind']=='hardware' and value['status']=='ok'):
            raise ValueError('Live energy replay mismatch')
    elif 'energy' in result or 'power.json' in files:
        raise ValueError('Unexpected live energy evidence')
    if result['correctness'].get('native_archive')=='native':
        from .live_native import validate
        from .integration_runner import runtime_observation
        checked=validate(root/'native')
        if dict(checked,native_archive='native')!=result['correctness']:raise ValueError('Native validation mismatch')
        native=json.loads((root/'native/record.json').read_text())
        rows=[json.loads(s) for s in native['stdout'].splitlines()]
        measured=[r for r in rows if r.get('kind')=='iteration' and not r['warmup']]
        case=json.loads((root/'native/plan.json').read_text())['case']
        if len(measured)!=len(result['records']):raise ValueError('Native/live record count mismatch')
        for native_row,entry in zip(measured,result['records']):
            if runtime_observation(native_row,controller.config,case)!=entry['observation']:
                raise ValueError('Live observation not bound to native work')
    elif result['evidence_kind']=='hardware':raise ValueError('Missing native correctness evidence')
    return result
