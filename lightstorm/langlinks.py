"""Πείραμα 6: ζευγάρια άρθρων ελληνικής ↔ Simple English Wikipedia, από τα interlanguage links.

Διαβάζει μόνο το δημόσιο API της el.wikipedia (generator=allpages + langlinks προς simple).
Γράφει το corpus_wiki/el_simple_links.tsv (ελληνικός τίτλος \\t αγγλικός τίτλος).

    python3 -m lightstorm.langlinks
"""
import json
import os
import subprocess
import time
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = "https://el.wikipedia.org/w/api.php"
UA = "TopIc-research/0.1 (Lightstorm experiment; github.com/ppetrakis86-tech/Top.Ic)"


def query(params):
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(8):
        try:
            # curl αντί για urllib: το API απαντούσε 429 σε κάθε αίτημα του urllib, ενώ στο curl 200
            out = subprocess.run(["curl", "-sS", "-f", "-m", "60", "-A", UA, url],
                                 capture_output=True, check=True).stdout
            return json.loads(out)
        except Exception as e:                     # 429 ή δίκτυο: αναμονή 5, 10, 20 ... έως 120 s
            if attempt == 7:
                raise
            wait = min(120, 5 * 2 ** attempt)
            print(f"σφάλμα {e}, ξαναδοκιμάζω σε {wait}s", flush=True)
            time.sleep(wait)


def main():
    out = os.path.join(ROOT, "corpus_wiki", "el_simple_links.tsv")
    params = {"action": "query", "format": "json", "formatversion": 2, "generator": "allpages",
              "gapnamespace": 0, "gapfilterredir": "nonredirects", "gaplimit": 500,
              "prop": "langlinks", "lllang": "simple", "lllimit": 500}
    pairs, n_req, cont = {}, 0, {}
    while True:
        data = query({**params, **cont})
        n_req += 1
        for page in data.get("query", {}).get("pages", []):
            for ll in page.get("langlinks", []):
                pairs[page["title"]] = ll["title"]
        if n_req % 50 == 0:
            print(f"{n_req} αιτήματα · {len(pairs):,} ζευγάρια", flush=True)
        if "continue" not in data:
            break
        cont = data["continue"]
        time.sleep(1.0)                            # ευγενικό ρυθμό στο API
    with open(out, "w", encoding="utf-8") as f:
        for el, en in sorted(pairs.items()):
            f.write(f"{el}\t{en}\n")
    print(f"✅ {len(pairs):,} ζευγάρια → {out} ({n_req} αιτήματα)")


if __name__ == "__main__":
    main()
