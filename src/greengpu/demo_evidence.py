"""Read-only operating system observations and per-step evidence for the demo."""

import csv
import hashlib
import io
import math
import os
from pathlib import Path
import subprocess
import threading
import time


def import_points(path):
    with path.open('rb') as stream:
        source = stream.read(8 * 1024 * 1024 + 1)
    if len(source) > 8 * 1024 * 1024:
        raise ValueError('Choose a CSV file under 8 MiB.')
    reader = csv.DictReader(io.StringIO(source.decode('utf-8-sig')), strict=True)
    if reader.fieldnames != ['id', 'x', 'y']:
        raise ValueError('The CSV first line must be id,x,y.')
    ids, points, seen = [], [], set()
    for row in reader:
        if len(points) == 64000:
            raise ValueError('Use at most 64,000 locations.')
        identity = (row['id'] or '').strip()
        if not identity or len(identity) > 128 or identity in seen or None in row:
            raise ValueError(f'CSV line {reader.line_num}: use a unique ID of up to 128 characters and exactly two coordinates.')
        try:
            point = [float(row['x']), float(row['y'])]
            if any(not math.isfinite(value) or abs(value) > 1e9 for value in point):
                raise ValueError()
        except (ValueError, TypeError):
            raise ValueError(f'CSV line {reader.line_num}: x and y must be numbers between -1 billion and 1 billion.') from None
        ids.append(identity)
        seen.add(identity)
        points.append(point)
    unique = sorted(set(map(tuple, points)))
    if len(unique) < 4:
        raise ValueError('Provide at least four distinct locations.')
    centers = [unique[i * (len(unique) - 1) // 3] for i in range(4)]
    return ids, points, centers, hashlib.sha256(source).hexdigest()


def _read_cpu(pid):
    fields = Path('/proc/stat').read_text().splitlines()[0].split()[1:9]
    counts = list(map(int, fields))
    system = (sum(counts), sum(counts[3:5]))
    process = None
    if pid is not None:
        try:
            stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
            process = (int(stat[19]), int(stat[11]) + int(stat[12]), int(stat[21]) * os.sysconf('SC_PAGE_SIZE'))
        except (OSError, ValueError, IndexError):
            pass
    return system, process


GPU_FIELDS = ('uuid', 'name', 'driver_version', 'pci.bus_id', 'utilization.gpu',
              'memory.used', 'temperature.gpu', 'clocks.current.graphics', 'power.draw')


def read_gpus():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=' + ','.join(GPU_FIELDS),
                                 '--format=csv,noheader,nounits'], capture_output=True,
                                text=True, timeout=1, check=True)
        devices = []
        for row in csv.reader(io.StringIO(result.stdout), skipinitialspace=True):
            if len(row) != len(GPU_FIELDS):
                continue
            device = dict(zip(GPU_FIELDS[:4], row[:4]))
            for field, value in zip(GPU_FIELDS[4:], row[4:]):
                try:
                    number = float(value)
                    device[field] = number if math.isfinite(number) and number >= 0 and (
                        field != 'utilization.gpu' or number <= 100) else None
                except ValueError:
                    device[field] = None
            devices.append(device)
        return devices, None if devices else 'No GPU readings returned.'
    except (OSError, subprocess.SubprocessError) as error:
        return [], f'GPU readings unavailable ({type(error).__name__}).'


class Monitor:
    """Samples separately from the controller. A short trial can yield no process sample."""

    def __init__(self, publish):
        self.publish = publish
        self.target = (None, None)
        self.error = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=3)
        if self.thread.is_alive():
            raise RuntimeError('Machine monitoring did not stop.')

    def _run(self):
        try:
            self._sample()
        except Exception as error:
            self.error = f'Machine sample collection stopped ({type(error).__name__}).'

    def _sample(self):
        previous = None
        while not self.stop_event.is_set():
            trial, pid = self.target
            acquired = time.monotonic_ns()
            host_percent = process_percent = memory = None
            try:
                system, process = _read_cpu(pid)
                if process:
                    memory = process[2]
                if previous:
                    old_host, old_process, old_pid, old_time = previous
                    total = system[0] - old_host[0]
                    if total > 0:
                        host_percent = 100 * (total - system[1] + old_host[1]) / total
                    if process and old_process and pid == old_pid and process[0] == old_process[0]:
                        ticks = process[1] - old_process[1]
                        process_percent = 100 * ticks / os.sysconf('SC_CLK_TCK') / ((acquired - old_time) / 1e9)
                previous = system, process, pid, acquired
                cpu_note = None
            except (OSError, ValueError, IndexError):
                cpu_note = 'Linux CPU counters unavailable.'
            gpu_start = time.monotonic_ns()
            gpus, gpu_note = read_gpus()
            self.publish(dict(utc_ns=time.time_ns(), cpu_sample_ns=acquired,
                              gpu_query_start_ns=gpu_start, gpu_query_end_ns=time.monotonic_ns(),
                              trial=trial, linux_pid=pid, linux_cpu_percent=host_percent,
                              native_cpu_percent_one_core=process_percent, native_memory_bytes=memory,
                              cpu_note=cpu_note, gpus=gpus, gpu_note=gpu_note))
            self.stop_event.wait(.35)


def step_evidence(row, barrier, size):
    def duration(start, end):
        a, b = row.get(start), row.get(end)
        return (b - a) / 1e6 if a is not None and b is not None else None

    cpu = row['cpu_items']
    decision = barrier.get('decision', {})
    observation = barrier.get('input', {}).get('observation', {})
    return dict(step=row['index'] + 1, release_monotonic_ns=row.get('release_ns'),
                cpu_items=cpu, gpu_items=row.get('gpu_items', size - cpu),
                cpu_work_ms=duration('cpu_start_ns', 'cpu_end_ns'),
                gpu_host_ms=duration('gpu_dispatch_ns', 'gpu_end_ns'),
                step_ms=duration('release_ns', 'end_ns'),
                merge_ms=duration('barrier_end_ns', 'end_ns'),
                input_upload_bytes=row.get('h2d_point_bytes'),
                center_upload_bytes=row.get('h2d_centroid_bytes'),
                labels_download_bytes=row.get('d2h_label_bytes'),
                aggregates_download_bytes=row.get('d2h_aggregate_bytes'),
                moved_items=row.get('ownership_changed_points'),
                decision=decision.get('allocation_reason'),
                next_cpu_items=decision.get('next_cpu_items'),
                cpu_completion_ms=observation.get('t_cpu', None) * 1000 if observation.get('t_cpu') is not None else None,
                gpu_completion_ms=observation.get('t_gpu', None) * 1000 if observation.get('t_gpu') is not None else None)
