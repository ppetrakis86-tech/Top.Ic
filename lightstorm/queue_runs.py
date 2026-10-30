"""Πείραμα 13: ουρά runs στην ίδια κάρτα με όριο VRAM. Ξεκινά το επόμενο run όταν χωράει, ξαναδοκιμάζει
(με συνέχεια από checkpoint) όποιο σταμάτησε χωρίς να τελειώσει, και γράφει queue_done.json στο τέλος.

    python -m lightstorm.queue_runs --jobs jobs.json --state DIR
jobs.json: [{"out": ".../s0", "arm": "A", "gb": 3.2, "args": [...]}, ...] (args: ό,τι παίρνει το lightstorm.train)
"""
import argparse
import json
import os
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


def done(job):
    return os.path.exists(os.path.join(job["out"], job["arm"], "epoch1.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--state", required=True, help="φάκελος για queue.log και queue_done.json")
    ap.add_argument("--budget", type=float, default=0, help="GB· 0 = όλη η κάρτα − 4")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--poll", type=float, default=30)
    a = ap.parse_args()
    jobs = json.load(open(a.jobs))
    budget = a.budget or gpu_total_gb() - 4
    log = open(os.path.join(a.state, "queue.log"), "a", encoding="utf-8")
    say = lambda m: (log.write(f"{time.strftime('%H:%M:%S')} {m}\n"), log.flush())
    say(f"ουρά: {len(jobs)} runs · όριο {budget:.1f} GB")
    running, tries, failed = {}, {i: 0 for i in range(len(jobs))}, set()
    while True:
        for i, p in list(running.items()):
            if p.poll() is not None:
                del running[i]
                j = jobs[i]
                say(f"τέλος {j['out']}/{j['arm']} · κωδικός {p.returncode} · {'✅' if done(j) else '❌'}")
                if not done(j) and tries[i] > a.retries:
                    failed.add(i)
        used = sum(jobs[i]["gb"] for i in running)
        for i, j in enumerate(jobs):
            if i in running or i in failed or done(j):
                continue
            if running and used + j["gb"] > budget:
                break                                  # με τη σειρά: τα μεγάλα πρώτα, κανείς δεν προσπερνά
            os.makedirs(j["out"], exist_ok=True)
            cmd = [sys.executable, "-m", "lightstorm.train", "--arm", j["arm"], "--out", j["out"],
                   "--trace", os.path.join(j["out"], f"trace_{j['arm']}.tsv")] + j["args"]
            running[i] = subprocess.Popen(cmd, stdout=open(os.path.join(j["out"], f"log_{j['arm']}.txt"), "a"),
                                          stderr=subprocess.STDOUT)
            tries[i] += 1
            used += j["gb"]
            say(f"ξεκινά {j['out']}/{j['arm']} (προσπάθεια {tries[i]}) · {used:.1f}/{budget:.1f} GB")
            time.sleep(20)                             # να πιάσει τη μνήμη του πριν το επόμενο
        if not running:
            break
        time.sleep(a.poll)
    res = {"done": sum(done(j) for j in jobs), "total": len(jobs),
           "failed": [f"{jobs[i]['out']}/{jobs[i]['arm']}" for i in sorted(failed)]}
    say(f"ΤΕΛΟΣ ουράς: {res}")
    json.dump(res, open(os.path.join(a.state, "queue_done.json"), "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
