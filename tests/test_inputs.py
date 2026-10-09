"""Regression tests for sampled and callable inputs in the Caputo solver."""

import math
import unittest
import warnings

import torch

from FDEint import FDEint


class TestInputAwareFDEint(unittest.TestCase):
    dtype = torch.float64

    def tensor(self, values, **kwargs):
        return torch.tensor(values, dtype=self.dtype, **kwargs)

    def assert_tensor_close(self, actual, expected, atol=1e-10, rtol=1e-10):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    @staticmethod
    def input_rhs(time, state, inputs):
        return inputs.expand_as(state)

    def test_analytic_constant_and_linear_forcing(self):
        time = torch.linspace(0, 1, 17, dtype=self.dtype)
        initial = self.tensor([2.0, -0.5])
        for alpha in (0.4, 0.7, 1.0):
            for power in (0, 1):
                with self.subTest(alpha=alpha, power=power):
                    solution = FDEint(
                        self.input_rhs, time, initial, alpha,
                        h=1 / 32, dtype=self.dtype, u=time ** power,
                    )
                    integral = (
                        math.gamma(power + 1) / math.gamma(power + alpha + 1)
                        * time ** (power + alpha)
                    )
                    expected = initial[None, None, :] + integral[None, :, None]
                    self.assert_tensor_close(solution, expected)

    def test_sampled_inputs_match_callback_on_nonuniform_grid(self):
        time = self.tensor([0.0, 0.0625, 0.25, 0.5, 1.0])
        initial = self.tensor([[0.2, -0.1], [1.0, 2.0]])

        def callback(query):
            return torch.cat((1 + 2 * query, -0.5 + query), dim=-1)

        samples = callback(time[:, None])
        sampled = FDEint(
            self.input_rhs, time, initial, 0.7,
            h=1 / 32, dtype=self.dtype, u=samples,
        )
        continuous = FDEint(
            self.input_rhs, time, initial, 0.7,
            h=1 / 32, dtype=self.dtype, u_fn=callback,
        )
        self.assert_tensor_close(sampled, continuous)

    def test_corrected_states_are_used_in_pece_history(self):
        # Two PECE steps for D y = lambda*y have this exact discrete result.
        # A lagged state stored under the next time gives 1.2769 at step two.
        time = self.tensor([0.0, 0.1, 0.2])
        solution = FDEint(
            lambda time, state: 1.3 * state,
            time, self.tensor([1.0]), 1.0, h=0.1, dtype=self.dtype,
        )
        self.assert_tensor_close(
            solution[0, :, 0], self.tensor([1.0, 1.13845, 1.2960684025]),
        )

    def test_shifted_initial_time(self):
        time = torch.linspace(2, 3, 9, dtype=self.dtype)
        alpha = 0.6
        solution, internal_time, _ = FDEint(
            self.input_rhs, time, self.tensor([3.0]), alpha,
            h=0.125, dtype=self.dtype, u_fn=lambda query: torch.ones_like(query),
            return_internals=True,
        )
        expected = 3 + (time - 2) ** alpha / math.gamma(alpha + 1)
        self.assert_tensor_close(solution[0, :, 0], expected)
        self.assert_tensor_close(internal_time[0], time)

    def test_float_and_tensor_step_sizes_match(self):
        time = torch.linspace(0, 1, 9, dtype=self.dtype)
        args = (self.input_rhs, time, self.tensor([0.0]), 0.7)
        float_step = FDEint(*args, h=0.125, dtype=self.dtype, u=time)
        tensor_step = FDEint(*args, h=self.tensor(0.125), dtype=self.dtype, u=time)
        default_step = FDEint(*args, dtype=self.dtype, u=time)
        self.assert_tensor_close(float_step, tensor_step)
        self.assert_tensor_close(float_step, default_step)

    def test_shared_and_batched_time_shapes(self):
        time = torch.linspace(0, 1, 9, dtype=self.dtype)
        initial = self.tensor([[0.0, 1.0], [2.0, 3.0]])
        alpha = 0.7
        expected = initial[:, None, :] + (
            time ** (alpha + 1) / math.gamma(alpha + 2)
        )[None, :, None]
        time_variants = (
            time,
            time[:, None],
            time[None, :].repeat(2, 1),
            time[None, :, None].repeat(2, 1, 1),
        )
        for times in time_variants:
            with self.subTest(shape=tuple(times.shape)):
                solution = FDEint(
                    self.input_rhs, times, initial, alpha,
                    dtype=self.dtype, u=time,
                )
                self.assert_tensor_close(solution, expected)

    def test_sample_shapes_preserve_batch_and_input_channels(self):
        time = torch.linspace(0, 1, 9, dtype=self.dtype)
        initial = self.tensor([[1.0, -1.0], [2.0, -2.0]])
        alpha = 0.4
        integral = time ** (alpha + 1) / math.gamma(alpha + 2)
        scalar_inputs = (
            time,
            time[:, None],
            time[None, :],
            time[None, :, None],
            time[None, :].repeat(2, 1),
            time[None, :, None].repeat(2, 1, 1),
        )
        for samples in scalar_inputs:
            with self.subTest(shape=tuple(samples.shape)):
                solution = FDEint(
                    self.input_rhs, time, initial, alpha,
                    dtype=self.dtype, u=samples,
                )
                self.assert_tensor_close(
                    solution, initial[:, None, :] + integral[None, :, None],
                )

        coefficients = self.tensor([[1.0, -2.0], [3.0, 4.0]])
        samples = time[None, :, None] * coefficients[:, None, :]
        batched = FDEint(
            self.input_rhs, time, initial, alpha, dtype=self.dtype, u=samples,
        )
        expected = initial[:, None, :] + integral[None, :, None] * coefficients[:, None, :]
        self.assert_tensor_close(batched, expected)

        shared = FDEint(
            self.input_rhs, time, initial, alpha, dtype=self.dtype, u=samples[0],
        )
        expected_shared = initial[:, None, :] + integral[None, :, None] * coefficients[0]
        self.assert_tensor_close(shared, expected_shared)

    def test_different_time_grids_per_batch(self):
        time = self.tensor([
            [0.0, 0.125, 0.25, 0.5, 1.0],
            [0.0, 0.25, 0.5, 0.75, 1.0],
        ])
        initial = self.tensor([[1.0], [2.0]])
        slopes = self.tensor([[1.0], [3.0]])
        samples = time * slopes
        alpha = 0.7
        expected = initial[:, None, :] + (
            time ** (alpha + 1) * slopes / math.gamma(alpha + 2)
        ).unsqueeze(-1)
        for times in (time, time.unsqueeze(-1)):
            with self.subTest(shape=tuple(times.shape)):
                solution = FDEint(
                    self.input_rhs, times, initial, alpha,
                    h=0.125, dtype=self.dtype, u=samples,
                )
                self.assert_tensor_close(solution, expected)

    def test_large_step_warns_and_reduces_to_requested_spacing(self):
        time = torch.linspace(0, 1, 9, dtype=self.dtype)
        args = (self.input_rhs, time, self.tensor([0.0]), 0.6)
        with self.assertWarnsRegex(UserWarning, "reducing h"):
            reduced = FDEint(*args, h=0.25, dtype=self.dtype, u=time)
        reference = FDEint(*args, h=0.125, dtype=self.dtype, u=time)
        self.assert_tensor_close(reduced, reference)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            FDEint(*args, h=0.125, dtype=self.dtype, u=time)
        self.assertEqual(caught, [])

    def test_exact_endpoint_does_not_add_an_extra_step(self):
        time = torch.linspace(0, 1, 9, dtype=self.dtype)
        evaluated_times = []

        def callback(query):
            evaluated_times.append(query.detach().clone())
            return torch.ones_like(query)

        solution, internal_time, internal_state = FDEint(
            self.input_rhs, time, self.tensor([0.0]), 0.7,
            h=0.125, dtype=self.dtype, u_fn=callback, return_internals=True,
        )
        self.assertEqual(tuple(solution.shape), (1, 9, 1))
        self.assertEqual(tuple(internal_state.shape), (1, 9, 1))
        self.assert_tensor_close(internal_time[0], time)
        self.assertEqual(len(evaluated_times), 9)
        self.assert_tensor_close(torch.cat(evaluated_times).flatten(), time)

    def test_rounded_uniform_grid_does_not_evaluate_past_endpoint(self):
        time = torch.linspace(0, 1, 11, dtype=self.dtype)

        def bounded_input(query):
            self.assertLessEqual(query.max().item(), 1.0)
            return torch.ones_like(query)

        for step in (None, 0.2):
            with self.subTest(step=step):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    _, internal_time, _ = FDEint(
                        self.input_rhs, time, self.tensor([0.0]), 0.7,
                        h=step, dtype=self.dtype, u_fn=bounded_input,
                        return_internals=True,
                    )
                self.assertEqual(internal_time.shape[1], 11)
                self.assert_tensor_close(internal_time[0], time)

    def test_sampled_inputs_hold_endpoint_on_partial_last_step(self):
        time = self.tensor([0.0, 0.3, 1.0])
        solution, internal_time, internal_state = FDEint(
            self.input_rhs, time, self.tensor([0.0]), 1.0,
            h=0.3, dtype=self.dtype, u=time, return_internals=True,
        )
        self.assert_tensor_close(internal_time[0, -1], self.tensor(1.2))
        self.assert_tensor_close(internal_state[0, -1, 0], self.tensor(0.69))
        self.assert_tensor_close(solution[0, -1, 0], self.tensor(0.5))

    def test_rejects_invalid_parameters_and_shapes(self):
        time = self.tensor([0.0, 0.5, 1.0])
        initial = self.tensor([0.0])
        base = {"f": self.input_rhs, "t": time, "y0": initial,
                "alpha": 0.7, "dtype": self.dtype, "u": time}
        invalid = (
            ("alpha_zero", {"alpha": 0.0}),
            ("alpha_negative", {"alpha": -0.1}),
            ("alpha_large", {"alpha": 1.1}),
            ("alpha_nan", {"alpha": float("nan")}),
            ("alpha_inf", {"alpha": float("inf")}),
            ("alpha_vector", {"alpha": self.tensor([0.4, 0.7])}),
            ("h_zero", {"h": 0.0}),
            ("h_negative", {"h": -0.1}),
            ("h_nan", {"h": float("nan")}),
            ("h_inf", {"h": float("inf")}),
            ("h_vector", {"h": self.tensor([0.1, 0.2])}),
            ("time_empty", {"t": self.tensor([])}),
            ("time_single", {"t": self.tensor([0.0])}),
            ("time_duplicate", {"t": self.tensor([0.0, 0.0, 1.0])}),
            ("time_descending", {"t": self.tensor([0.0, 1.0, 0.5])}),
            ("time_nan", {"t": self.tensor([0.0, float("nan"), 1.0])}),
            ("time_wrong_rank", {"t": torch.zeros(1, 1, 3, 1, dtype=self.dtype)}),
            ("time_wrong_channels", {"t": torch.zeros(1, 3, 2, dtype=self.dtype)}),
            ("time_batch_mismatch", {"t": time[None, :].repeat(2, 1)}),
            ("different_initial_times", {
                "t": self.tensor([[0.0, 0.5, 1.0], [0.1, 0.6, 1.1]]),
                "y0": self.tensor([[0.0], [0.0]]),
            }),
            ("initial_empty", {"y0": self.tensor([])}),
            ("initial_scalar", {"y0": self.tensor(0.0)}),
            ("initial_wrong_rank", {"y0": torch.zeros(1, 1, 1, dtype=self.dtype)}),
            ("initial_nan", {"y0": self.tensor([float("nan")])}),
            ("input_wrong_length", {"u": self.tensor([0.0, 1.0])}),
            ("input_wrong_batch", {"u": torch.zeros(2, 3, 1, dtype=self.dtype)}),
            ("input_wrong_rank", {"u": torch.zeros(1, 1, 3, 1, dtype=self.dtype)}),
            ("input_no_channels", {"u": torch.zeros(3, 0, dtype=self.dtype)}),
            ("input_nan", {"u": self.tensor([0.0, float("nan"), 1.0])}),
            ("unsupported_dtype", {"dtype": torch.float16}),
            ("callback_wrong_rank", {"u": None, "u_fn": lambda query: query.flatten()}),
            ("callback_wrong_batch", {"u": None, "u_fn": lambda query: query.repeat(2, 1)}),
            ("callback_non_tensor", {"u": None, "u_fn": lambda query: 1.0}),
            ("rhs_wrong_shape", {"f": lambda time, state, inputs: state.flatten()}),
            ("rhs_non_tensor", {"f": lambda time, state, inputs: 0.0}),
        )
        for name, replacement in invalid:
            with self.subTest(case=name):
                arguments = dict(base, **replacement)
                with self.assertRaises(ValueError):
                    FDEint(**arguments)

        invalid_types = (
            {"t": [0.0, 0.5, 1.0]},
            {"y0": [0.0]},
            {"u": [0.0, 0.5, 1.0]},
            {"f": None},
            {"u": None, "u_fn": 1.0},
        )
        for replacement in invalid_types:
            with self.subTest(types=replacement.keys()):
                with self.assertRaises(TypeError):
                    FDEint(**dict(base, **replacement))

    def test_rejects_unrepresentable_step_before_grid_allocation(self):
        for dtype, start, step in (
            (torch.float32, 1.0, 1e-40),
            (torch.float64, 1.0, 1e-320),
            (torch.float64, 1.0, 1e-20),
            (torch.float64, 0.0, 1e-320),
        ):
            with self.subTest(dtype=dtype, start=start, step=step):
                time = torch.tensor([start, start + 1.0], dtype=dtype)
                with self.assertRaisesRegex(ValueError, "h is too small"):
                    FDEint(
                        lambda time, state: -state, time,
                        torch.tensor([1.0], dtype=dtype), 0.7,
                        h=step, dtype=dtype,
                    )

    def test_sampled_and_callable_inputs_are_mutually_exclusive(self):
        time = self.tensor([0.0, 0.5, 1.0])
        with self.assertRaisesRegex(ValueError, "either u or u_fn"):
            FDEint(
                self.input_rhs, time, self.tensor([0.0]), 0.7,
                u=time, u_fn=lambda query: query, dtype=self.dtype,
            )

    def test_gradients_for_order_initial_state_rhs_and_samples(self):
        time = self.tensor([0.0, 0.125, 0.25, 0.375])
        alpha = self.tensor(0.65, requires_grad=True)
        coefficient = self.tensor(0.15, requires_grad=True)
        initial = self.tensor([0.5], requires_grad=True)
        samples = self.tensor([0.1, 0.2, 0.3, 0.4], requires_grad=True)

        def solve(order, weight, state, inputs):
            return FDEint(
                lambda time, state, inputs: weight * state + inputs,
                time, state, order, u=inputs, dtype=self.dtype,
            )

        self.assertTrue(torch.autograd.gradcheck(
            solve, (alpha, coefficient, initial, samples),
            eps=1e-6, atol=1e-5, rtol=1e-4,
        ))
        # Training RHS and inputs must also work when alpha is fixed.
        self.assertTrue(torch.autograd.gradcheck(
            lambda weight, state, inputs: solve(alpha.detach(), weight, state, inputs),
            (coefficient, initial, samples), eps=1e-6, atol=1e-5, rtol=1e-4,
        ))

    def test_gradients_for_input_callback_parameter(self):
        time = self.tensor([0.0, 0.125, 0.25, 0.375])
        weight = self.tensor(0.3, requires_grad=True)

        def solve(parameter):
            return FDEint(
                self.input_rhs, time, self.tensor([0.5]), 0.65,
                u_fn=lambda query: parameter * (1 + query), dtype=self.dtype,
            )

        self.assertTrue(torch.autograd.gradcheck(
            solve, (weight,), eps=1e-6, atol=1e-5, rtol=1e-4,
        ))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cpu_controls_and_samples_follow_cuda_initial_state(self):
        time = torch.linspace(0, 1, 9, dtype=self.dtype)
        alpha = self.tensor(0.7)
        samples = time.clone()
        initial = self.tensor([0.5]).cuda()
        solution = FDEint(
            self.input_rhs, time, initial, alpha,
            h=self.tensor(0.125), dtype=self.dtype, u=samples,
        )
        self.assertEqual(solution.device, initial.device)
        expected = 0.5 + time ** 1.7 / math.gamma(2.7)
        self.assert_tensor_close(solution[0, :, 0].cpu(), expected)


if __name__ == "__main__":
    unittest.main()
