"""Unit tests for the run-spec receipt (eval_harness/run_spec.py).

No model loading: build_run_spec is a pure function of the config. Research
configs rebuild the REAL door objects model-free (build_doors), which is the
same path the sweep and the live run share.
"""
import dataclasses
import unittest

from eval_harness.config import EvalConfig
from eval_harness import run_spec as rs


def _cfg(**over):
    base = dict(benchmark="longbench", subsets="qasper,narrativeqa", backend="research",
                model="meta-llama/Llama-3.1-8B-Instruct", temperature=0.0, max_requests=200)
    base.update(over)
    return EvalConfig(**base)


def _research_cfg(kv="ridge", ratio=0.9, kv_kwargs=None, **over):
    """An EvalConfig for a research run — build_run_spec rebuilds the REAL door
    objects from this (model-free), exactly like the live run does."""
    llm = {"attn_implementation": "sdpa",
           "research_config": {"kv_compressor": kv, "compression_ratio": ratio,
                               "kv_compressor_kwargs": dict(kv_kwargs or {})}}
    return _cfg(backend="research", llm_kwargs=llm, **over)


# --- Tests -----------------------------------------------------------------
class TestRunSpecBasics(unittest.TestCase):
    def test_plain_run_has_empty_method_block_and_stable_fp(self):
        spec = rs.build_run_spec(_cfg(backend="hf"))
        self.assertEqual(spec["method"],
                         {"positional": None, "attention": None, "kv_compressor": None})
        self.assertEqual(len(spec["fingerprint"]), 16)
        # Deterministic: same inputs -> same barcode.
        self.assertEqual(spec["fingerprint"],
                         rs.build_run_spec(_cfg(backend="hf"))["fingerprint"])

    def test_fingerprint_flips_on_setting_change_only(self):
        base = rs.build_run_spec(_cfg())["fingerprint"]
        self.assertNotEqual(base, rs.build_run_spec(_cfg(temperature=0.7))["fingerprint"])
        self.assertNotEqual(base, rs.build_run_spec(_cfg(max_requests=500))["fingerprint"])
        # output_dir is noise -> must NOT change the barcode.
        self.assertEqual(base, rs.build_run_spec(_cfg(output_dir="/somewhere/else"))["fingerprint"])

    def test_subsets_order_independent(self):
        a = rs.build_run_spec(_cfg(subsets="qasper,narrativeqa"))["fingerprint"]
        b = rs.build_run_spec(_cfg(subsets="narrativeqa,qasper"))["fingerprint"]
        self.assertEqual(a, b)


class TestMethodCapture(unittest.TestCase):
    """Exercises the REAL door builders (build_doors) via a research config —
    the model-free path the sweep and the live run share."""

    def _ridge_knobs(self, **kv_over):
        spec = rs.build_run_spec(_research_cfg(kv="ridge", kv_kwargs=kv_over))
        return spec["method"]["kv_compressor"]["knobs"]

    def test_defaults_captured_even_when_unset(self):
        knobs = self._ridge_knobs()
        self.assertEqual(knobs["sink_size"], 8)      # never set by user, still recorded
        self.assertEqual(knobs["local_size"], 64)
        self.assertEqual(knobs["envelope_gamma"], 1.0)
        self.assertEqual(knobs["ridge_lambda"], 1e-4)

    def test_method_scoped_no_cross_leak(self):
        knobs = self._ridge_knobs()
        self.assertIn("envelope_gamma", knobs)       # ridge-specific
        self.assertNotIn("chunk_size", knobs)        # compactor-specific, absent

    def test_knob_change_flips_fingerprint(self):
        a = rs.build_run_spec(_research_cfg(kv="ridge"))["fingerprint"]
        b = rs.build_run_spec(_research_cfg(kv="ridge", kv_kwargs={"envelope_gamma": 2.0}))["fingerprint"]
        self.assertNotEqual(a, b)

    def test_ratio_change_flips_fingerprint(self):
        a = rs.build_run_spec(_research_cfg(kv="ridge", ratio=0.9))["fingerprint"]
        b = rs.build_run_spec(_research_cfg(kv="ridge", ratio=0.6))["fingerprint"]
        self.assertNotEqual(a, b)

    def test_different_method_different_knobs(self):
        knobs = rs.build_run_spec(_research_cfg(kv="compactor"))["method"]["kv_compressor"]["knobs"]
        self.assertIn("chunk_size", knobs)
        self.assertNotIn("envelope_gamma", knobs)

    def test_deterministic_no_drift(self):
        # Same config -> same barcode, twice. This is the runner==sweep guarantee
        # (both call this same function on the same config).
        a = rs.build_run_spec(_research_cfg(kv="ridge", kv_kwargs={"envelope_gamma": 2.0}))
        b = rs.build_run_spec(_research_cfg(kv="ridge", kv_kwargs={"envelope_gamma": 2.0}))
        self.assertEqual(a["fingerprint"], b["fingerprint"])


class TestVersions(unittest.TestCase):
    def test_versions_block_scoped_and_present(self):
        v = rs.build_run_spec(_research_cfg(kv="ridge"))["versions"]
        self.assertEqual(v["framework"], rs.FRAMEWORK_VERSION)
        self.assertIn("longbench", v.get("benchmark", {}))          # active benchmark
        self.assertIn("ridge", v.get("kv_compressor", {}))          # active method
        self.assertNotIn("attention_method", v)                     # none active -> absent

    def test_component_version_bump_flips_barcode(self):
        from eval_harness.kv_compression import get_kv_compressor_class
        ridge_cls = get_kv_compressor_class("ridge")
        before = rs.build_run_spec(_research_cfg(kv="ridge"))["fingerprint"]
        old = ridge_cls.VERSION
        try:
            ridge_cls.VERSION = old + 1
            after = rs.build_run_spec(_research_cfg(kv="ridge"))["fingerprint"]
        finally:
            ridge_cls.VERSION = old
        self.assertNotEqual(before, after)                          # bump -> new barcode

    def test_framework_version_bump_flips_barcode(self):
        before = rs.build_run_spec(_cfg(backend="hf"))["fingerprint"]
        old = rs.FRAMEWORK_VERSION
        try:
            rs.FRAMEWORK_VERSION = old + 1
            after = rs.build_run_spec(_cfg(backend="hf"))["fingerprint"]
        finally:
            rs.FRAMEWORK_VERSION = old
        self.assertNotEqual(before, after)

    def test_version_is_not_a_knob(self):
        knobs = rs.build_run_spec(_research_cfg(kv="ridge"))["method"]["kv_compressor"]["knobs"]
        self.assertNotIn("VERSION", knobs)                          # class attr, not a field

    def test_git_metadata_not_in_barcode(self):
        # code provenance is recorded but must not affect the fingerprint.
        spec = rs.build_run_spec(_cfg(backend="hf"))
        payload_fp = rs.fingerprint(spec)                           # recomputed excludes 'code'
        self.assertEqual(payload_fp, spec["fingerprint"])
        # If a git repo, 'code' is present with the two fields.
        if "code" in spec:
            self.assertIn("git_commit", spec["code"])
            self.assertIn("git_dirty", spec["code"])

    def test_schema_version_not_in_barcode(self):
        spec = rs.build_run_spec(_cfg(backend="hf"))
        fp0 = spec["fingerprint"]
        spec["spec_schema_version"] = 999
        self.assertEqual(rs.fingerprint(spec), fp0)                 # schema bump != rerun


class TestDumpDataclass(unittest.TestCase):
    def test_array_like_field_dropped(self):
        class _Arr:
            ndim = 2

        @dataclasses.dataclass
        class _M:
            keep: int = 5
            drop: object = None

        m = _M()
        m.drop = _Arr()
        out = rs._dump_dataclass(m)
        self.assertEqual(out["keep"], 5)
        self.assertNotIn("drop", out)


class TestDoneMarker(unittest.TestCase):
    def test_intended_is_cap_times_subsets(self):
        m = rs.build_done_marker(
            fingerprint="abc", actual_samples=3150, overall_score=48.86,
            max_requests=200,
            requested_subsets=["qasper", "narrativeqa"] + [f"s{i}" for i in range(14)],
            per_subset_actual={"qasper": 200, "narrativeqa": 200},
            loaded_before_cap=9000)
        self.assertEqual(m["samples"]["intended"], 200 * 16)
        self.assertTrue(m["samples"]["short"])          # 3150 < 3200
        self.assertEqual(m["samples"]["actual"], 3150)
        self.assertEqual(m["overall_score"], 48.86)
        self.assertTrue(m["complete"])
        self.assertEqual(m["fingerprint"], "abc")

    def test_not_short_when_cap_met(self):
        m = rs.build_done_marker(
            fingerprint="abc", actual_samples=400, max_requests=200,
            requested_subsets=["a", "b"],
            per_subset_actual={"a": 200, "b": 200}, loaded_before_cap=5000)
        self.assertEqual(m["samples"]["intended"], 400)
        self.assertFalse(m["samples"]["short"])

    def test_per_subset_dict_sums(self):
        m = rs.build_done_marker(
            fingerprint="x", actual_samples=150,
            max_requests_per_subset={"a": 100, "b": 100},
            requested_subsets=["a", "b"], per_subset_actual={"a": 100, "b": 50},
            loaded_before_cap=1000)
        self.assertEqual(m["samples"]["intended"], 200)
        self.assertTrue(m["samples"]["short"])          # b came up 50 short

    def test_uncapped_intended_is_loaded(self):
        m = rs.build_done_marker(fingerprint="x", actual_samples=42,
                                 loaded_before_cap=42)
        self.assertEqual(m["samples"]["intended"], 42)
        self.assertFalse(m["samples"]["short"])

    def test_numpy_counts_are_json_safe(self):
        import json
        m = rs.build_done_marker(
            fingerprint="x", actual_samples=3,
            per_subset_actual={"a": _np_int(3)}, max_requests=3,
            requested_subsets=["a"], loaded_before_cap=3)
        json.dumps(m)  # must not raise
        self.assertEqual(m["samples"]["per_subset_actual"]["a"], 3)


def _np_int(v):
    try:
        import numpy as np
        return np.int64(v)
    except Exception:
        return v


class TestValueNormalization(unittest.TestCase):
    def test_arbitrary_object_skipped(self):
        self.assertIs(rs._jsonable(object()), rs._SKIP)

    def test_array_like_skipped(self):
        class _Arr:
            ndim = 2
        self.assertIs(rs._jsonable(_Arr()), rs._SKIP)

    def test_zero_d_scalar_unwrapped(self):
        class _Scalar:
            ndim = 0
            def item(self):
                return 3.5
        self.assertEqual(rs._jsonable(_Scalar()), 3.5)


if __name__ == "__main__":
    unittest.main()
