import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from noise import GaussianNoise, build_noise, derive_seed, validate_noise


class NoiseTests(unittest.TestCase):
    def test_zero_is_identity_and_does_not_advance_any_rng(self):
        noise = GaussianNoise(0, 42)
        hidden = torch.ones(1, 2, 32, dtype=torch.float16)
        local, global_state = noise.generator.get_state().clone(), torch.get_rng_state().clone()
        self.assertIs(noise.apply(hidden), hidden)
        self.assertTrue(torch.equal(local, noise.generator.get_state()))
        self.assertTrue(torch.equal(global_state, torch.get_rng_state()))
        self.assertEqual(noise.describe()["effective_noise_rms"], 0.0)

    def test_private_seed_reproducibility_and_global_rng_isolation(self):
        hidden = torch.zeros(1, 20, 32)
        state = torch.get_rng_state().clone()
        a, b, c = GaussianNoise(0.2, 42), GaussianNoise(0.2, 42), GaussianNoise(0.2, 43)
        first = a.apply(hidden)
        self.assertTrue(torch.equal(first, b.apply(hidden)))
        self.assertFalse(torch.equal(first, c.apply(hidden)))
        self.assertFalse(torch.equal(first, a.apply(hidden)))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_gaussian_absolute_standard_deviation(self):
        result = GaussianNoise(0.5, 9).apply(torch.zeros(1, 1000, 128))
        self.assertLess(abs(result.mean().item()), 0.006)
        self.assertAlmostEqual(result.std().item(), 0.5, delta=0.006)

    def test_preserves_tensor_contract_and_input(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            hidden = torch.ones(1, 7, 32, dtype=dtype)
            old = hidden.clone()
            result = GaussianNoise(0.2, 42).apply(hidden)
            self.assertEqual(result.shape, hidden.shape)
            self.assertEqual(result.dtype, dtype)
            self.assertEqual(result.device, hidden.device)
            self.assertNotEqual(result.data_ptr(), hidden.data_ptr())
            self.assertTrue(torch.equal(hidden, old))

    def test_reports_effective_rounded_noise(self):
        hidden = torch.full((1, 2, 32), 1000.0, dtype=torch.float16)
        noise = GaussianNoise(1e-8, 42)
        result = noise.apply(hidden)
        self.assertTrue(torch.equal(hidden, result))
        report = noise.describe()
        self.assertEqual(report["effective_noise_rms"], 0)
        self.assertEqual(report["changed_element_fraction"], 0)
        self.assertEqual(report["clean_activation_rms"], 1000)

    def test_stats_match_actual_delta(self):
        hidden = torch.linspace(-1, 1, 64).reshape(1, 2, 32).half()
        noise = GaussianNoise(0.1, 42)
        result = noise.apply(hidden)
        delta = result.double() - hidden.double()
        report = noise.describe()
        self.assertAlmostEqual(report["effective_noise_rms"], delta.square().mean().sqrt().item())
        self.assertAlmostEqual(report["effective_mean_abs_noise"], delta.abs().mean().item())
        self.assertEqual(report["elements"], 64)
        self.assertEqual(report["forward_calls"], 1)

    def test_invalid_config(self):
        valid = {"distribution": "gaussian", "sigma": 0.02, "seed": 42}
        for extra in (
            {"sigma": -1}, {"sigma": True}, {"sigma": float("inf")}, {"sigma": float("nan")},
            {"seed": True}, {"seed": -1}, {"seed": 2**63}, {"distribution": "unregistered"}, {"other": 1},
        ):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                validate_noise({**valid, **extra})
        self.assertIsInstance(build_noise(valid), GaussianNoise)

    def test_invalid_tensor(self):
        for hidden in (
            torch.zeros(2, 3, 32), torch.zeros(1, 0, 32), torch.zeros(1, 32),
            torch.zeros(1, 2, 32, dtype=torch.long), torch.full((1, 2, 32), float("nan")),
        ):
            with self.assertRaises(ValueError):
                GaussianNoise(0, 42).apply(hidden)

    def test_nonfinite_rounded_output_rejected(self):
        with self.assertRaisesRegex(ValueError, "Nothing was uploaded"):
            GaussianNoise(1e10, 42).apply(torch.ones(1, 2, 32, dtype=torch.float16))

    def test_case_seeds_are_stable_and_distinct(self):
        self.assertEqual(derive_seed(42, "case"), derive_seed(42, "case"))
        self.assertNotEqual(derive_seed(42, "case"), derive_seed(43, "case"))
        self.assertNotEqual(derive_seed(42, "case"), derive_seed(42, "other"))
        self.assertLess(derive_seed(42, "case"), 2**63)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable; CPU noise tests remain required.")
    def test_cuda_fp16_uses_private_cpu_samples_at_long_lengths(self):
        for length in (1, 128, 1024):
            hidden = torch.ones(1, length, 32, device="cuda", dtype=torch.float16)
            state = torch.cuda.get_rng_state().clone()
            result = GaussianNoise(0.1, 42).apply(hidden)
            expected = GaussianNoise(0.1, 42).apply(hidden.cpu())
            self.assertTrue(torch.equal(result.cpu(), expected))
            self.assertTrue(torch.equal(state, torch.cuda.get_rng_state()))
            self.assertEqual(result.device, hidden.device)


if __name__ == "__main__":
    unittest.main()
