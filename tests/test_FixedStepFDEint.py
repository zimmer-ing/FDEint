import torch
import unittest
import time
import numpy as np
import matplotlib.pyplot as plt
from FDEint.FixedStepFDEint import FDEint
from tests.mittag_leffler import ml as mittag_leffler

# Define the fractional differential equation as a simple function (dy/dt = -y)
def fractional_diff_eq(t, x):
    return -x

class TestFDEintSolver(unittest.TestCase):
    def setUp(self):
        # Define the data types for which the solver will be tested
        self.data_types = [torch.float32, torch.float64]
        self.num_steps = 1000
        self.results = {}

    def test_solver(self):
        for dtype in self.data_types:
            with self.subTest(dtype=dtype):
                print(f"Running solver with dtype: {dtype}")

                # Define the time points for the simulation
                t = torch.linspace(0., 20., self.num_steps + 1, dtype=dtype)

                # Real values for comparison using the Mittag-Leffler function
                real_values = [mittag_leffler(-i.item() ** 0.6, 0.6) for i in t]

                # Initial condition for the system (batch size 1)
                y0 = torch.tensor([1., 1.], dtype=dtype)

                # Start the timer
                start_time = time.time()

                # Prepare the initial condition for batch processing
                batch_size = 1
                y0_batch = y0.repeat(batch_size, 1)

                # Solve the fractional differential equation
                solver_values = FDEint(
                    fractional_diff_eq,
                    t,
                    y0_batch,
                    torch.tensor([0.6], dtype=dtype).unsqueeze(0),
                    h=torch.tensor(20 / self.num_steps, dtype=dtype),
                    dtype=dtype,
                    DEBUG=False
                )

                # End the timer
                end_time = time.time()

                # Print the time taken for the solver to run
                print(f'Time taken by solver: {end_time - start_time:.4f} seconds')

                # Compute the error between the solver's output and the real values
                real_values_np = np.array(real_values)
                solver_values_np = solver_values[0].detach().numpy()
                error = real_values_np.flatten() - solver_values_np[:, 0].flatten()

                # Check if the total error is within acceptable limits
                total_error = np.sum(np.abs(error)) / len(error)
                self.results[dtype] = {
                    'time': end_time - start_time,
                    'total_error': total_error
                }

                # Set the acceptable error threshold (can be adjusted)
                acceptable_error = 0.0009
                self.assertLess(total_error, acceptable_error, f"Total error for dtype {dtype} exceeded acceptable limit.")

                # Optional: display error plot
                plt.plot(t.detach().numpy(), error, label=f'Error (dtype={dtype})')
                plt.legend()
                plt.show()

    def tearDown(self):
        # Summarize the results after all tests
        print("Results summary:")
        for dtype, result in self.results.items():
            print(f"dtype: {dtype}, time taken: {result['time']:.4f} seconds, total error: {result['total_error']:.6f}")


if __name__ == '__main__':
    unittest.main()