"""Πείραμα 13: ουρά runs στην ίδια κάρτα με όριο VRAM. Ξεκινά το επόμενο run όταν χωράει, ξαναδοκιμάζει
(με συνέχεια από checkpoint) όποιο σταμάτησε χωρίς να τελειώσει, και γράφει queue_done.json στο τέλος.

    python -m lightstorm.queue_runs --jobs jobs.json --state DIR
jobs.json: [{"out": ".../s0", "arm": "A", "gb": 3.2, "args": [...]}, ...] (args: ό,τι παίρνει το lightstorm.train)
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time


def gpu_total_gb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout
        return float(out.split()[0]) / 1024
    except (OSError, ValueError, IndexError):
        return 0.0


def free_ram_gb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2 ** 20
    except OSError:
        pass
    return 1e9


def run_name(j):
    """Όνομα των τοπικών log/trace ενός run (το χρησιμοποιεί και ο πιλότος για τις μπάρες)."""
    return "_".join(j["out"].rstrip("/").split("/")[-2:]) + "_" + j["arm"]


def done(job):
    return os.path.exists(os.path.join(job["out"], job["arm"], "epoch1.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--state", required=True, help="φάκελος για queue.log και queue_done.json")
    ap.add_argument("--budget", type=float, default=0, help="GB· 0 = όλη η κάρτα − 4")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--logs", default=None, help="τοπικός φάκελος για log/trace (όχι στο Drive)· αντίγραφο στο out στο τέλος κάθε run")
    ap.add_argument("--min-ram", type=float, default=10, help="GB ελεύθερης RAM που πρέπει να μένουν για να μπει νέο run")
    ap.add_argument("--poll", type=float, default=30)
    a = ap.parse_args()
    jobs = json.load(open(a.jobs))
    budget = a.budget or gpu_total_gb() - 4
    logs = a.logs or a.state
    os.makedirs(logs, exist_ok=True)
    paths = [os.path.join(logs, "queue.log"), os.path.join(a.state, "queue.log")]
    env = dict(os.environ, PYTHONUNBUFFERED="1")

    def say(m):                                        # τοπικά πρώτα· αν το Drive σκαλώσει, η ουρά δεν πέφτει
        for p in dict.fromkeys(paths):
            try:
                with open(p, "a", encoding="utf-8") as f:
                    f.write(f"{time.strftime('%H:%M:%S')} {m}\n")
            except OSError:
                pass

    def keep_copy(j):
        for ext in ("log", "trace"):
            try:
                shutil.copy(os.path.join(logs, f"{run_name(j)}.{ext}"), os.path.join(j["out"], f"{ext}_{j['arm']}.txt"))
            except OSError:
                pass
    say(f"ουρά: {len(jobs)} runs · όριο {budget:.1f} GB")
    beat = 0
    running, tries, failed = {}, {i: 0 for i in range(len(jobs))}, set()
    while True:
        for i, p in list(running.items()):
            if p.poll() is not None:
                del running[i]
                j = jobs[i]
                keep_copy(j)
                say(f"τέλος {j['out']}/{j['arm']} · κωδικός {p.returncode} · {'✅' if done(j) else '❌'} · RAM ελεύθερη {free_ram_gb():.0f} GB")
                if not done(j) and tries[i] > a.retries:
                    failed.add(i)
        used = sum(jobs[i]["gb"] for i in running)
        for i, j in enumerate(jobs):
            if i in running or i in failed or done(j):
                continue
            if running and (used + j["gb"] > budget or free_ram_gb() < a.min_ram):
                break                                  # με τη σειρά: τα μεγάλα πρώτα, κανείς δεν προσπερνά
            os.makedirs(j["out"], exist_ok=True)
            cmd = [sys.executable, "-m", "lightstorm.train", "--arm", j["arm"], "--out", j["out"],
                   "--trace", os.path.join(logs, f"{run_name(j)}.trace")] + j["args"]
            running[i] = subprocess.Popen(cmd, stdout=open(os.path.join(logs, f"{run_name(j)}.log"), "a"),
                                          stderr=subprocess.STDOUT, env=env)
            tries[i] += 1
            used += j["gb"]
            say(f"ξεκινά {j['out']}/{j['arm']} (προσπάθεια {tries[i]}) · {used:.1f}/{budget:.1f} GB · RAM ελεύθερη {free_ram_gb():.0f} GB")
            time.sleep(20)                             # να πιάσει τη μνήμη του πριν το επόμενο
        if not running:
            break
        time.sleep(a.poll)
        beat += 1
        if beat % 20 == 0:                             # κάθε ~10 λεπτά: ζωντανή, με RAM
            say(f"τρέχουν {len(running)} · RAM ελεύθερη {free_ram_gb():.0f} GB")
    res = {"done": sum(done(j) for j in jobs), "total": len(jobs),
           "failed": [f"{jobs[i]['out']}/{jobs[i]['arm']}" for i in sorted(failed)]}
    say(f"ΤΕΛΟΣ ουράς: {res}")
    for p in dict.fromkeys([logs, a.state]):
        try:
            json.dump(res, open(os.path.join(p, "queue_done.json"), "w"), ensure_ascii=False)
        except OSError:
            pass


if __name__ == "__main__":
    main()
