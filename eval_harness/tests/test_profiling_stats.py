"""Statistics contract for the performance benchmark.

Every value here is hand-computed. These functions turn raw latencies into the
numbers that go in a paper table, so a silent convention change (population vs
sample stdev, p50-based vs total throughput) must fail loudly.
"""

from __future__ import annotations

import unittest

from eval_harness.profiling.stats import (
    LatencySummary, decode_throughput_median, decode_throughput_total, linfit,
    percentile, prefill_throughput, speedup, summarize,
)


class TestPercentile(unittest.TestCase):
    def test_linear_interpolation_convention(self):
        # numpy 'linear': pos = q/100 * (n-1); p95 of [1..5] -> pos 3.8 -> 4 + .8*(5-4)
        for q, want in ((0, 1.0), (50, 3.0), (90, 4.6), (95, 4.8), (99, 4.96), (100, 5.0)):
            with self.subTest(q=q):
                self.assertAlmostEqual(percentile([1, 2, 3, 4, 5], q), want)

    def test_two_and_one_element(self):
        self.assertAlmostEqual(percentile([10, 20], 95), 19.5)
        self.assertAlmostEqual(percentile([10, 20], 50), 15.0)
        for q in (0, 50, 99, 100):
            self.assertAlmostEqual(percentile([7.0], q), 7.0)

    def test_order_invariance(self):
        self.assertAlmostEqual(percentile([5, 1, 4, 2, 3], 95), percentile([1, 2, 3, 4, 5], 95))

    def test_rejects_empty_and_out_of_range(self):
        with self.assertRaises(ValueError):
            percentile([], 50)
        with self.assertRaises(ValueError):
            percentile([1, 2], 101)


class TestSummarize(unittest.TestCase):
    def test_fields_hand_computed(self):
        s = summarize([1, 2, 3, 4])
        self.assertIsInstance(s, LatencySummary)
        self.assertEqual(s.n, 4)
        self.assertAlmostEqual(s.mean, 2.5)
        self.assertAlmostEqual(s.median, 2.5)
        self.assertAlmostEqual(s.min, 1.0)
        self.assertAlmostEqual(s.max, 4.0)
        # SAMPLE stdev = sqrt(5/3); the population value would be sqrt(1.25).
        self.assertAlmostEqual(s.std, 1.2909944487358056)
        self.assertAlmostEqual(s.cv, s.std / s.mean)

    def test_empty_returns_none_not_zero(self):
        self.assertIsNone(summarize([]))

    def test_single_sample_has_zero_std(self):
        self.assertAlmostEqual(summarize([4.2]).std, 0.0)


class TestThroughput(unittest.TestCase):
    def test_the_two_conventions_are_different(self):
        """Total-time and median-based throughput must never be collapsed.

        [20,20,20,40] ms: total = 100 ms for 4 tokens -> 40 tok/s, but the median
        step is 20 ms -> 50 tok/s. Reporting the latter as "tok/s" overstates by
        25% by discarding exactly the tail where stalls live.
        """
        steps = [20.0, 20.0, 20.0, 40.0]
        self.assertAlmostEqual(decode_throughput_total(steps), 40.0)
        self.assertAlmostEqual(decode_throughput_median(steps), 50.0)
        self.assertNotAlmostEqual(decode_throughput_total(steps), decode_throughput_median(steps))

    def test_total_equals_reciprocal_of_mean(self):
        steps = [12.0, 13.0, 11.0, 14.0]
        self.assertAlmostEqual(decode_throughput_total(steps), 1000.0 / (sum(steps) / len(steps)))

    def test_prefill_throughput_exact(self):
        self.assertAlmostEqual(prefill_throughput(32768, 4096.0), 8000.0)

    def test_empty_and_zero(self):
        self.assertIsNone(decode_throughput_total([]))
        self.assertIsNone(prefill_throughput(100, 0.0))


class TestSpeedup(unittest.TestCase):
    def test_ratio(self):
        self.assertAlmostEqual(speedup(40.0, 20.0), 2.0)

    def test_missing_anchor_is_none_never_one(self):
        """A missing denominator must not render as 'no difference'."""
        self.assertIsNone(speedup(None, 20.0))
        self.assertIsNone(speedup(40.0, None))
        self.assertIsNone(speedup(40.0, 0.0))


class TestLinfit(unittest.TestCase):
    def test_exact_line(self):
        slope, intercept = linfit([1, 2, 3, 4], [3, 5, 7, 9])
        self.assertAlmostEqual(slope, 2.0)
        self.assertAlmostEqual(intercept, 1.0)

    def test_degenerate(self):
        self.assertIsNone(linfit([1], [2]))
        self.assertIsNone(linfit([1, 1], [2, 3]))


if __name__ == "__main__":
    unittest.main()
