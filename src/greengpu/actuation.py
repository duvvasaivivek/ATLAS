"""Transactional clock control: unknown prior policy blocks every setter.

The NVML adapter intentionally cannot certify an unqueryable prior locked-clock
policy. An API symbol or current observed frequency is not a policy snapshot.
"""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import signal
import tempfile
import threading
import time

from .experiment import encode


class ActuationBlocked(RuntimeError):
    pass


class RestorationFailed(RuntimeError):
    pass


def valid_policy(snapshot):
    if snapshot.get("restorable") is not True or not snapshot.get("device_uuid"):
        return False
    policy = snapshot.get("policy")
    if not isinstance(policy, dict) or set(policy) != {"core", "memory"}:
        return False
    for value in policy.values():
        # None means a *known* unlocked policy, never an unknown policy.
        if value is not None and (not isinstance(value, (tuple,list)) or len(value)!=2 or
                                 any(type(x) is not int or x<=0 for x in value) or value[0]>value[1]):
            return False
    return snapshot.get("policy_source") == "authoritative_policy_readback"


class ClockTransaction:
    """Exclusive experiment owner; durable intent before mutation, restore on exit.

    Backend contract: snapshot(), set_domain(domain, range_or_none), device_uuid.
    snapshot must distinguish known unlocked state from missing information.
    """

    def __init__(self, backend, directory, pairs):
        self.backend, self.directory = backend, Path(directory)
        self.pairs = {tuple(p) for p in pairs}
        if any(len(p)!=2 or any(type(x) is not int or x<=0 for x in p) for p in self.pairs):
            raise ValueError("Invalid explicit feasible pairs")
        self.prior = None
        self.dirty = False
        self.events = []
        self.stream = None
        self.lock_file = None
        self.previous_sigterm = None
        self.signal_installed = False

    def lock(self):
        # OS-released lock prevents separate ATLAS runs from restoring over one
        # another. External tools remain an explicitly uncontrolled interference.
        if os.name!="posix":
            raise ActuationBlocked("Clock transactions require the Linux/WSL process-lock adapter")
        import fcntl
        key=hashlib.sha256(self.backend.device_uuid.encode()).hexdigest()
        path=Path(tempfile.gettempdir())/("atlas-frequency-"+key+".lock")
        fd=os.open(path,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
        self.lock_file=os.fdopen(fd,"a+")
        try:fcntl.flock(self.lock_file.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock_file.close();self.lock_file=None
            raise ActuationBlocked("Another ATLAS clock transaction owns this GPU") from None

    def unlock(self):
        if self.lock_file is not None:self.lock_file.close();self.lock_file=None

    def record(self, kind, **fields):
        event = dict(sequence=len(self.events),kind=kind,monotonic_ns=time.monotonic_ns(),**fields)
        self.stream.write(encode(event)+"\n"); self.stream.flush(); os.fsync(self.stream.fileno())
        self.events.append(event)

    def __enter__(self):
        self.directory.mkdir(parents=True,exist_ok=False)
        self.stream=(self.directory/"journal.jsonl").open("x",encoding="utf-8",newline="\n")
        try:
            self.lock()
            self.prior=deepcopy(self.backend.snapshot())
            self.record("snapshot",snapshot=self.prior)
            if not valid_policy(self.prior) or self.prior["device_uuid"] != self.backend.device_uuid:
                self.record("blocked",reason="prior_policy_not_restorable")
                raise ActuationBlocked("Prior clock policy cannot be captured and verified")
            if threading.current_thread() is not threading.main_thread():
                raise ActuationBlocked("Clock transactions require the main thread for handled termination")
            commands=[]
            for domain,lock_flag,reset_flag in (("memory","-lmc","-rmc"),("core","-lgc","-rgc")):
                bounds=self.prior["policy"][domain]
                commands.append(["nvidia-smi","-i",self.backend.device_uuid]+([reset_flag] if bounds is None else [lock_flag,",".join(map(str,bounds))]))
            recovery=dict(device_uuid=self.backend.device_uuid,prior_policy=self.prior["policy"],
                          recovery_argv=commands,
                          instructions="Restore both recorded domains on this UUID, then verify policy readback. Never infer policy from observed clocks.")
            with (self.directory/"recovery.json").open("x",encoding="utf-8") as stream:
                stream.write(json.dumps(recovery,indent=2)+"\n"); stream.flush(); os.fsync(stream.fileno())
            self.previous_sigterm=signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM,self.handle_sigterm)
            self.signal_installed=True
            return self
        except BaseException:
            self.stream.close()
            self.unlock()
            raise

    @staticmethod
    def handle_sigterm(signum,frame):
        raise KeyboardInterrupt("SIGTERM during clock transaction")

    def apply(self, pair):
        pair=tuple(pair)
        if pair not in self.pairs:
            raise ActuationBlocked("Unlisted frequency pair")
        self.dirty=True  # Before any setter: even a failing call may have side effects.
        self.record("request",pair=list(pair),requested_pair=list(pair))
        for domain,value in (("memory",pair[1]),("core",pair[0])):
            try:
                self.backend.set_domain(domain,[value,value])
                self.record("api_acknowledged",domain=domain,range=[value,value],physical_verification=False)
            except BaseException as error:
                self.record("apply_failed",domain=domain,error=f"{type(error).__name__}: {error}")
                raise
        return self.events[-1]["monotonic_ns"]

    def restore(self):
        if not self.dirty:
            self.record("restoration",status="not_needed")
            return
        failures=[]
        for domain in ("memory","core"):
            try:
                self.record("restore_request",domain=domain,range=self.prior["policy"][domain])
            except BaseException as error:
                failures.append(f"journal: {type(error).__name__}: {error}")
            try:
                self.backend.set_domain(domain,self.prior["policy"][domain])
            except BaseException as error:
                failures.append(f"{domain}: {type(error).__name__}: {error}")
        try:
            after=self.backend.snapshot()
            if (not valid_policy(after) or after["device_uuid"]!=self.prior["device_uuid"] or
                    encode(after["policy"])!=encode(self.prior["policy"])):
                failures.append("policy_readback_mismatch")
        except BaseException as error:
            after=None; failures.append(f"readback: {type(error).__name__}: {error}")
        self.record("restoration",status="failed" if failures else "verified",failures=failures,snapshot=after)
        if failures:
            raise RestorationFailed("Clock restoration failed; consult recovery.json: "+"; ".join(failures))
        self.dirty=False

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.restore()
        finally:
            self.stream.close()
            self.unlock()
            if self.signal_installed:
                signal.signal(signal.SIGTERM,self.previous_sigterm)
                self.signal_installed=False
        return False


class NvmlClockBackend:
    def __init__(self, policy_reader=None):
        # A platform adapter may supply authoritative policy readback. This is
        # not exposed as a user-asserted baseline or inferred from clock rates.
        self.policy_reader=policy_reader
        import pynvml as nv
        self.nv=nv
        nv.nvmlInit()
        try:
            if nv.nvmlDeviceGetCount()!=1:
                raise RuntimeError("Exactly one visible GPU is required")
            self.handle=nv.nvmlDeviceGetHandleByIndex(0)
            self.device_uuid=nv.nvmlDeviceGetUUID(self.handle)
        except BaseException:
            nv.nvmlShutdown(); raise

    def query(self, function, *args):
        start=time.monotonic_ns()
        try:
            result=dict(status="ok",value=function(self.handle,*args))
        except self.nv.NVMLError as error:
            result=dict(status="unavailable",value=None,error=str(error),error_code=error.value)
        return dict(result,acquisition_start_ns=start,acquisition_end_ns=time.monotonic_ns())

    def snapshot(self):
        if self.policy_reader is not None:
            snapshot=self.policy_reader(self.device_uuid)
            if not valid_policy(snapshot) or snapshot["device_uuid"]!=self.device_uuid:
                raise ActuationBlocked("Invalid authoritative policy provider")
            return deepcopy(snapshot)
        n=self.nv
        return dict(device_uuid=self.device_uuid,restorable=False,policy=None,
                    policy_source="unavailable",reason="locked_clock_policy_getters_not_exposed",
                    observed_core=self.query(n.nvmlDeviceGetClockInfo,n.NVML_CLOCK_GRAPHICS),
                    observed_memory=self.query(n.nvmlDeviceGetClockInfo,n.NVML_CLOCK_MEM),
                    application_core=self.query(n.nvmlDeviceGetApplicationsClock,n.NVML_CLOCK_GRAPHICS),
                    application_memory=self.query(n.nvmlDeviceGetApplicationsClock,n.NVML_CLOCK_MEM),
                    default_application_core=self.query(n.nvmlDeviceGetDefaultApplicationsClock,n.NVML_CLOCK_GRAPHICS),
                    default_application_memory=self.query(n.nvmlDeviceGetDefaultApplicationsClock,n.NVML_CLOCK_MEM),
                    application_permission=self.query(n.nvmlDeviceGetAPIRestriction,n.NVML_RESTRICTED_API_SET_APPLICATION_CLOCKS))

    def discover(self):
        n=self.nv; pairs=[]; errors=[]
        try:
            for memory in n.nvmlDeviceGetSupportedMemoryClocks(self.handle):
                pairs.extend([core,memory] for core in n.nvmlDeviceGetSupportedGraphicsClocks(self.handle,memory))
        except n.NVMLError as error: errors.append(str(error))
        pci=n.nvmlDeviceGetPciInfo(self.handle).busId
        return dict(device_uuid=self.device_uuid,name=n.nvmlDeviceGetName(self.handle),
                    driver=n.nvmlSystemGetDriverVersion(),pci_bus_id=pci.decode() if isinstance(pci,bytes) else pci,
                    pairs=sorted(pairs),pair_provenance="driver_reported_unverified",errors=errors,
                    prior=self.snapshot(),locked_core="blocked_unknown_prior_policy",locked_memory="blocked_unknown_prior_policy",
                    application_clocks="not_used_deprecated_and_not_equivalent_to_locked_policy",
                    power_cap="not_an_independent_frequency_actuator",settings_changed=False)

    def set_domain(self, domain, bounds):
        if domain not in ("core","memory"):
            raise ValueError("Unknown clock domain")
        if bounds is not None and (len(bounds)!=2 or any(type(x) is not int or x<=0 for x in bounds) or bounds[0]>bounds[1]):
            raise ValueError("Invalid locked-clock bounds")
        if not valid_policy(self.snapshot()):
            raise ActuationBlocked("NVML locked-clock prior policy is unqueryable; no setter issued")
        n=self.nv
        setter=n.nvmlDeviceSetGpuLockedClocks if domain=="core" else n.nvmlDeviceSetMemoryLockedClocks
        reset=n.nvmlDeviceResetGpuLockedClocks if domain=="core" else n.nvmlDeviceResetMemoryLockedClocks
        if bounds is None: reset(self.handle)
        else: setter(self.handle,*bounds)

    def close(self):
        self.nv.nvmlShutdown()
