"""Own NVML in a spawned process so a driver query cannot strand the runner."""
import multiprocessing as mp
import math
import time


def loaded_factory():
    from .actuation import NvmlClockBackend
    from .frequency_probe import LoadedSampler
    return LoadedSampler(NvmlClockBackend())


def telemetry_factory():
    from .telemetry import NvmlTelemetry
    return NvmlTelemetry()


def _worker(connection, factory):
    sampler = None
    try:
        sampler = factory()
        connection.send((True, None))  # Initialization is separate from a query.
        while True:
            method = connection.recv()
            if method == "stop":
                break
            if method not in ("sample", "counter"):
                raise ValueError("Unknown sampling request")
            start = time.thread_time_ns()
            frame = getattr(sampler, method)()
            if method == "sample":
                frame["poll_cpu_ns"] = time.thread_time_ns() - start
            connection.send((True, frame))
    except EOFError:
        pass
    except BaseException as error:
        try:
            connection.send((False, f"{type(error).__name__}: {error}"))
        except (BrokenPipeError, OSError):
            pass
    finally:
        # Parent termination is bounded even if shutdown blocks in the driver.
        if sampler is not None:
            backend = getattr(sampler, "backend", sampler)
            if hasattr(backend, "close"):
                backend.close()
        connection.close()


class IsolatedSampler:
    def __init__(self, factory=loaded_factory, timeout=2, startup_timeout=10):
        if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0
               for v in (timeout,startup_timeout)):
            raise ValueError('Sampler deadlines must be finite and positive')
        self.timeout = timeout
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_worker, args=(child, factory), daemon=True)
        self.process.start()
        child.close()
        self.closed = False
        try:
            self._receive(startup_timeout, 'initialization')
        except BaseException:
            self.close()
            raise

    def _receive(self, timeout, stage):
        if not self.connection.poll(timeout):
            self.close()
            raise TimeoutError(f'Telemetry {stage} deadline exceeded')
        try:
            okay, value = self.connection.recv()
        except EOFError:
            self.close()
            raise RuntimeError(f'Telemetry worker exited during {stage}') from None
        if not okay:
            raise RuntimeError(value)
        return value

    def query(self, method):
        if self.closed:
            raise RuntimeError("Sampler is closed")
        self.connection.send(method)
        return self._receive(self.timeout, 'query')

    def sample(self):
        return self.query("sample")

    def counter(self):
        return self.query("counter")

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(.5)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(.5)
        self.connection.close()
        if self.process.is_alive():
            raise RuntimeError("Telemetry process could not be reaped")
