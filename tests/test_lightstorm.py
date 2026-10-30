"""Γ1·4: δοκιμές πριν από το training. Τρέχουν σε CPU σε λίγα δευτερόλεπτα.

    python3 -m unittest -v tests.test_lightstorm
"""
import unittest

import numpy as np
import torch

from lightstorm import metrics
from lightstorm.data import is_sealed, split_blocks
from lightstorm.model import ARMS, Config, Lightstorm, count_params, schedule


class TestSheet(unittest.TestCase):
    def test_torus(self):
        D = metrics.torus_distances()
        self.assertTrue(torch.equal(D, D.T))
        self.assertAlmostEqual(D.max().item(), 32 ** 0.5, places=5)    # το πιο μακρινό: (4,4)
        cr = metrics.cross()
        for i in range(64):                                            # όλοι έχουν 4 γείτονες σε απόσταση 1
            self.assertEqual(sorted(D[i, cr[i]].tolist()), [0, 1, 1, 1, 1])

    def test_schedule(self):
        self.assertAlmostEqual(schedule(0, 100)[0], 2.5)
        self.assertAlmostEqual(schedule(100, 100)[0], 0.7)
        self.assertAlmostEqual(schedule(250, 100)[0], 0.7)            # 2η φάση: σταθερό
        self.assertAlmostEqual(schedule(250, 100)[1], 0.01)


class TestMetrics(unittest.TestCase):
    def test_usage(self):
        w = np.zeros(64); w[:30] = 100
        m = w * 5; m[30:60] = 100
        u = metrics.usage(w, m, 3000)
        self.assertEqual((u["winners"], u["neighbours"], u["idle"]), (30, 30, 4))
        self.assertFalse(u["idle_alarm"])
        self.assertTrue(metrics.usage(w, w, 3000)["idle_alarm"])

    def test_gold_perfect_and_random(self):
        D = metrics.torus_distances().numpy()
        home = np.full(100, -1); pairs = []
        for i in range(40):
            home[i] = home[50 + i] = i % 64
            pairs.append((i, 50 + i))
        self.assertEqual(metrics.gold_distance(home, pairs, D)["gold"], 0.0)
        rng = np.random.default_rng(0)
        home[50:90] = rng.integers(0, 64, 40)
        self.assertGreater(metrics.gold_distance(home, pairs, D)["ratio"], 0.5)

    def test_tokens_to_target(self):
        self.assertEqual(metrics.tokens_to_target([(10, 5.0), (20, 4.0), (30, 3.0)], 4.0), 20)
        self.assertIsNone(metrics.tokens_to_target([(10, 5.0)], 4.0))


class TestData(unittest.TestCase):
    def test_split(self):
        ids = np.arange(20480 * 3, dtype=np.uint16)
        tr, va = split_blocks(ids)
        self.assertEqual(len(tr) + len(va), len(ids))
        self.assertAlmostEqual(len(va) / len(ids), 0.1, delta=0.01)
        self.assertFalse(set(tr.tolist()) & set(va.tolist()))

    def test_seal_deterministic(self):
        pairs = [(f"λέξη{i}", f"leksi{i}") for i in range(1000)]
        flags = [is_sealed(*p) for p in pairs]
        self.assertEqual(flags, [is_sealed(*p) for p in pairs])
        self.assertAlmostEqual(sum(flags) / 1000, 0.5, delta=0.06)


class TestModel(unittest.TestCase):
    def test_params(self):
        for arm in ARMS:
            n = count_params(Lightstorm(Config(arm=arm)))
            self.assertTrue(7.0e6 < n < 7.6e6, (arm, n))

    def test_keys_get_no_gradient(self):
        m = Lightstorm(Config(arm="B"))
        self.assertNotIn("keys", [n.split(".")[-1] for n, _ in m.named_parameters()])

    def test_som_moves_keys_and_orders_sheet(self):
        """Ο κανόνας SOM κινεί τα κλειδιά, και με μεγάλο σ οι γείτονες στο φύλλο μοιάζουν περισσότερο."""
        torch.manual_seed(0)
        moe = Lightstorm(Config(arm="B")).moe
        before = moe.keys.clone()
        moe.train()
        x = torch.randn(4096, 256)
        for step in range(60):
            moe.sigma, moe.eta = schedule(step, 60, eta=(0.5, 0.05))
            moe(x[torch.randint(0, 4096, (512,))])
            moe.som_step()
        self.assertGreater((moe.keys - before).norm().item(), 1.0)
        kd = torch.cdist(moe.keys, moe.keys)
        D = metrics.torus_distances()
        near, far = kd[D == 1].mean(), kd[D >= 3].mean()
        self.assertLess(near, far)

    def test_shuffle_changes_output_only_with_topology(self):
        torch.manual_seed(0)
        m = Lightstorm(Config(arm="B")).eval()
        m.moe.sigma = 0.7
        x = torch.randint(0, 14679, (2, 64))
        base, _, _ = m(x)
        m.moe.set_permutation(torch.arange(64))
        same, _, _ = m(x)
        self.assertTrue(torch.allclose(base, same))
        perm = metrics.shuffle_permutations()[0]
        m.moe.set_permutation(perm)
        D = m.moe.D
        self.assertTrue(torch.equal(D, D.T))
        for i in range(64):                        # οι νέοι γείτονες είναι σε απόσταση 1 στις νέες θέσεις
            self.assertEqual(sorted(D[i, m.moe.cross[i]].tolist()), [0, 1, 1, 1, 1])
        moved, _, _ = m(x)
        self.assertFalse(torch.allclose(base, moved))

    def test_ema_frozen_in_eval(self):
        m = Lightstorm(Config(arm="C1")).eval()
        before = m.moe.ema.clone()
        m(torch.randint(0, 14679, (2, 32)))
        m.moe.som_step()
        self.assertTrue(torch.equal(before, m.moe.ema))

    def test_overfit_one_batch(self):
        """Κάθε σκέλος πρέπει να μάθει απ' έξω ένα μικρό batch (από ~9,6 κάτω από 1).

        Τα B/C μαθαίνουν πιο αργά και με θόρυβο: όσο κινούνται τα κλειδιά, αλλάζει ο νικητής.
        """
        for arm in ARMS:
            torch.manual_seed(0)
            m = Lightstorm(Config(arm=arm))
            if arm != "A":
                m.moe.sigma, m.moe.eta = 0.7, 0.05
            opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
            x = torch.randint(0, 14679, (2, 33))
            for _ in range(200):
                _, loss, aux = m(x[:, :-1], x[:, 1:])
                opt.zero_grad(); loss.backward(); opt.step()
                if arm != "A":
                    m.moe.som_step()
            self.assertLess(aux["ce"].item(), 1.0, arm)


if __name__ == "__main__":
    unittest.main()


class TestSubset(unittest.TestCase):
    """Πείραμα 13: υποσύνολο του train ανά πηγή."""
    def setUp(self):
        import json, tempfile
        self.sizes = {"wiki_el": (40000, 300), "wiki_en": (30000, 200), "greeklish": (20000, 100), "clean": (10500, 50)}
        self.train = torch.cat([torch.full((t,), i) for i, (t, _) in enumerate(self.sizes.values())])
        self.val = torch.cat([torch.full((v,), i) for i, (_, v) in enumerate(self.sizes.values())])
        self.man = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False).name
        json.dump({"sources": {**{k: {"train_tokens": t, "val_tokens": v} for k, (t, v) in self.sizes.items()},
                               "clean_dedup_dropped_chars": 5}}, open(self.man, "w"))

    def test_full_is_identity(self):
        from lightstorm.train import subset
        tr, va, segs = subset(self.train, self.val, self.man)
        self.assertTrue(torch.equal(tr, self.train) and torch.equal(va, self.val))
        self.assertEqual([k for k, _ in segs], list(self.sizes))

    def test_frac_and_sources(self):
        from lightstorm.train import subset
        tr, va, segs = subset(self.train, self.val, self.man, frac=0.25, block=1000)
        self.assertAlmostEqual(len(tr) / len(self.train), 0.25, delta=0.03)
        for i, (t, _) in enumerate(self.sizes.values()):      # κάθε πηγή στο ίδιο ποσοστό
            self.assertAlmostEqual((tr == i).sum().item() / t, 0.25, delta=0.06)
        keep = ["wiki_el", "wiki_en"]
        tr2, va2, segs2 = subset(self.train, self.val, self.man, keep, frac=0.25, block=1000)
        self.assertAlmostEqual(len(tr2) / len(tr), 1, delta=0.05)          # ίδιο πλήθος tokens, άλλες πηγές
        self.assertEqual(set(tr2.unique().tolist()), {0, 1})
        self.assertEqual(len(va2), 500)
        self.assertEqual(segs2, [("wiki_el", 300), ("wiki_en", 200)])
        _, va3, _ = subset(None, self.val, self.man, keep)
        self.assertTrue(torch.equal(va3, va2))
        with self.assertRaises(AssertionError):
            subset(self.train, self.val, self.man, ["clean"], frac=0.25)   # δεν φτάνουν
