"""Bounded Linux native-process supervision, including inherited output pipes."""
import codecs
import os
import selectors
import signal
import subprocess
import tempfile
import threading
import time


def terminate_group(process):
    # The group can outlive its leader. poll() must not guard killpg().
    if process is not None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def capture_command(argv, timeout):
    """Capture a noninteractive command without waiting on descendant-owned pipes."""
    if os.name!='posix':
        return subprocess.run(argv,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout,check=False)
    with tempfile.TemporaryFile() as output,tempfile.TemporaryFile() as errors:
        process=subprocess.Popen(argv,stdout=output,stderr=errors,start_new_session=True)
        expired=False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            expired=True
        finally:
            terminate_group(process)
            process.wait(timeout=2)
        output.seek(0);errors.seek(0)
        stdout=output.read().decode('utf-8',errors='replace')
        stderr=errors.read().decode('utf-8',errors='replace')
        if expired:raise subprocess.TimeoutExpired(argv,timeout,output=stdout,stderr=stderr)
        return subprocess.CompletedProcess(argv,process.returncode,stdout,stderr)


class Supervisor:
    def __init__(self, argv, stderr, timeout=120, failure=None):
        if os.name != "posix":
            raise RuntimeError("Native supervision requires Linux/WSL")
        if timeout <= 0:
            raise ValueError("Positive process timeout required")
        self.failure = failure or (lambda: None)
        self.expired = threading.Event()
        self.process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=stderr, text=True, encoding="utf-8",
                                        bufsize=1, start_new_session=True)
        self.timer = threading.Timer(timeout, self._expire)
        self.timer.daemon = True
        self.timer.start()
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.decoder = codecs.getincrementaldecoder("utf-8")()
        self.buffer = ""
        self.eof = False

    def _expire(self):
        self.expired.set()
        terminate_group(self.process)

    def check(self):
        error = self.failure()
        if error:
            terminate_group(self.process)
            raise RuntimeError(error)
        if self.expired.is_set():
            raise TimeoutError("Native process deadline exceeded")

    def receive(self, timeout=None):
        deadline = time.monotonic() + timeout if timeout is not None else None
        while True:
            self.check()
            if "\n" in self.buffer:
                line, self.buffer = self.buffer.split("\n", 1)
                return line + "\n"
            if self.eof:
                line, self.buffer = self.buffer, ""
                return line or None
            if deadline is not None and time.monotonic() >= deadline:
                self._expire()
                self.check()
            if self.selector.select(.02):
                raw = os.read(self.process.stdout.fileno(), 65536)
                self.buffer += self.decoder.decode(raw, final=not raw)
                self.eof = not raw
                if len(self.buffer) > 32 * 1024 * 1024:
                    raise ValueError("Native protocol record exceeds 32 MiB")

    def lines(self):
        while True:
            line = self.receive()
            if line is None:
                return
            yield line

    def close(self):
        self.timer.cancel()
        terminate_group(self.process)
        try:
            self.process.wait(timeout=2)
        finally:
            self.selector.close()
            self.process.stdin.close()
            self.process.stdout.close()
