"""Fresh, non-actuating qualification. Never infer policy or repair driver time."""
import json
from pathlib import Path
from .experiment import save_new
from .frequency_experiment import sha


def clock_diagnostic(frames):
    ages=[f['wall_end_us']-s['timestamp_us'] for f in frames for d in ('core','memory') for s in f['buffer_samples'][d]]
    offsets=[f['wall_end_us']-f['acquisition_end_ns']/1000 for f in frames]
    return dict(samples=len(ages),future_samples=sum(a<0 for a in ages),
                minimum_age_us=min(ages) if ages else None,maximum_age_us=max(ages) if ages else None,
                host_wall_monotonic_offset_span_us=max(offsets)-min(offsets) if offsets else None,
                status='rejected_future_driver_time' if any(a<0 for a in ages) else 'no_samples' if not ages else 'no_future_samples_observed',
                offset_fitted=False,live_qualified=False)


def summarize(root,diagnostic,measurement):
    root=Path(root)
    manifest=json.loads((root/'loaded/manifest.json').read_text())
    frames=[frame for name in manifest['trials'] for frame in json.loads((root/'loaded'/name).read_text())['frames']]
    capability=json.loads((root/'loaded/capabilities.json').read_text())
    return dict(clock_mapping=clock_diagnostic(frames),actuation_gate=diagnostic['hardware_exit_gate'],
                prior_policy=capability['prior'],energy_qualification=measurement['energy_qualification'],
                device_timing_qualification=measurement['device_timing_qualification'],
                cpu_interfaces=capability['cpu_interfaces'],wall_energy='unavailable_no_external_meter',
                live_joint_ready=False,settings_changed=False,
                next_requirements=['authoritative restorable prior clock policy',
                                   'qualified host/driver sample timing',
                                   'independently qualified energy domain before savings claims'])


def replay(root):
    from .phase8 import replay as replay_loaded
    from .runner import replay as replay_measurement
    root=Path(root);expected=json.loads((root/'checksums.json').read_text())
    files={p.relative_to(root).as_posix():p for p in root.rglob('*') if p.is_file() and p!=root/'checksums.json'}
    if set(files)!=set(expected) or any(p.is_symlink() or sha(p)!=expected[name] for name,p in files.items()):
        raise ValueError('Qualification archive integrity failure')
    result=summarize(root,replay_loaded(root/'loaded'),replay_measurement(root/'measurement'))
    if result!=json.loads((root/'summary.json').read_text()):raise ValueError('Qualification summary mismatch')
    return result


def run(output,build,blocks=3):
    from .phase8 import run as diagnostics
    from .runner import run_calibration
    root=Path(output);root.mkdir(parents=True,exist_ok=False);build=Path(build)
    diagnostic=diagnostics(root/'loaded',build/'atlas_frequency_probe',blocks,2.)
    measurement=run_calibration(root/'measurement',build/'atlas_measure',blocks,10.,.1)
    result=summarize(root,diagnostic,measurement)
    save_new(root/'summary.json',result)
    save_new(root/'checksums.json',{p.relative_to(root).as_posix():sha(p) for p in sorted(root.rglob('*')) if p.is_file()})
    replay(root)
    return result
