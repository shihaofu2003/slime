import concurrent.futures
import os
import threading
import time


def evaluate_domain(payload, domain):
    if os.getpid() == payload["parent_pid"]:
        raise AssertionError("parallel domain evaluation ran in the parent process")
    if domain == payload.get("fail_domain"):
        raise RuntimeError(f"fixture failure: {domain}")
    time.sleep(payload.get("domain_delays", {}).get(domain, 0.0))
    return {
        "domain": domain,
        "pid": os.getpid(),
        "domain_concurrency": payload.get("_domain_concurrency"),
    }


def evaluate_borrowing(payload, domain):
    if os.getpid() == payload["parent_pid"]:
        raise AssertionError("parallel domain evaluation ran in the parent process")
    if domain == "airline":
        return {"domain": domain, "max_active": 0}

    slot_semaphore = payload["_slot_semaphore"]
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    active = 0
    max_active = 0

    def run_probe():
        nonlocal active, max_active
        slot_semaphore.acquire()
        try:
            with lock:
                active += 1
                max_active = max(max_active, active)
            barrier.wait(timeout=5)
        finally:
            with lock:
                active -= 1
            slot_semaphore.release()

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=payload["_executor_max_workers"]
    ) as executor:
        futures = [executor.submit(run_probe) for _ in range(2)]
        for future in futures:
            future.result()
    return {"domain": domain, "max_active": max_active}
