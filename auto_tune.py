import numpy as np
import concurrent.futures
import subprocess
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import plotly.io as py
import plotly.graph_objects as go

CONC_POPULATION = 10  # Max number of concurrent populations, GPU resource bound
START_POPULATIONS = 2000  # Max number of total spawned populations
STEPSIZE = 1e-3  # Step size for gradient descent
LEARNING_RATE = 1e-6  # Learning rate for gradient descent
EPOCHS = 10  # Max epochs per population
BATCH_SIZE_STAGES = [(5, 0.1), (50, 0.1), (1000, 0.05), (20000, 1)] # Batch size and survival rate for the next generation

optimal_population_numbers = []
all_pids = []
all_costs = []


# Get total cost from tinyphysics simulator, by piping the process output from ProcessPoolExecutor to be more performent than using the class with ThreadPoolExecutor. Lilely due to bottlenecks in current setup of onnix runtime.
def run_simulation(pid):
    p, i, d = pid
    result = subprocess.run(
        [
            "python3",
            "tinyphysics.py",
            "--model_path",
            "./models/tinyphysics.onnx",
            "--data_path",
            "./data",
            "--num_segs",
            str(BATCH_SIZE),
            "--controller",
            "pid",
            "--no-graph",
            "--kp",
            str(p),
            "--ki",
            str(i),
            "--kd",
            str(d),
        ],
        capture_output=True,
        text=True,
    )
    output = result.stdout
    parts = output.split(",")
    total_cost = float("inf")
    for part in parts:
        if "average sum_total_cost" in part and ":" in part:
            total_cost = float(part.split(":")[1].strip())
            break
    return total_cost


# Gradient descent algorithm
def gradient_descent(population, lr, initial_guess):
    total_epochs = 0
    params = initial_guess
    costs = []

    # In case of overshooting, we want to store the lowest cost found.
    lowest_cost = float("inf")
    lowest_cost_pid = None
    epoch = 0

    previous_cost = None
    for epoch in range(EPOCHS):
        if total_epochs >= EPOCHS:
            break
        cost = run_simulation(params)
        costs.append(cost)

        # print(f'Population {population}, Epoch {total_epochs}, Cost: {cost}, PID: {params}')
        if cost < lowest_cost:
            lowest_cost = cost
            lowest_cost_pid = params.copy()
        total_epochs += 1

        # Conduct inital topolgy search to warm start, using low batch size
        if (
            batch_size == BATCH_SIZE_STAGES[0][0]
            or batch_size == BATCH_SIZE_STAGES[3][0]
        ):
            break

        # Kill a population if it's not improving
        if previous_cost is not None:
            rate_of_change = (cost - previous_cost) / previous_cost
            if rate_of_change > 0 or abs(rate_of_change) < 0.001:
                break

        previous_cost = cost

        gradients = np.zeros_like(params)
        delta = STEPSIZE

        # Calculate gradients
        for i in range(len(params)):
            params[i] += delta
            cost_plus = run_simulation(params)
            params[i] -= 2 * delta
            cost_minus = run_simulation(params)
            params[i] += delta
            gradients[i] = (cost_plus - cost_minus) / (2 * delta)

        # Update PID parameters
        params -= lr * gradients

    return params, costs, lowest_cost, lowest_cost_pid, epoch


def optimize_pid(initial_guess, population):

    lr = LEARNING_RATE
    best_pid, costs, lowest_cost, lowest_cost_pid, epoch = gradient_descent(
        population, lr, initial_guess
    )
    return best_pid, costs, lowest_cost, lowest_cost_pid, epoch


# Set initial guess for PID parameters for a population
def generate_all_initial_guesses(
    total_populations=START_POPULATIONS,
    lower_bounds=[0, 0, 0],
    upper_bounds=[0.02, 0.3, 0.15],
):
    divisions = round(total_populations ** (1 / 3))
    step_sizes = [
        (upper - lower) / divisions for lower, upper in zip(lower_bounds, upper_bounds)
    ]
    initial_guesses = [
        np.array([i * step_sizes[0], j * step_sizes[1], k * step_sizes[2]])
        for i in range(divisions)
        for j in range(divisions)
        for k in range(divisions)
    ]

    return initial_guesses


all_initial_guesses = generate_all_initial_guesses()

for stage, (batch_size, top_percentage) in enumerate(BATCH_SIZE_STAGES):
    BATCH_SIZE = batch_size  # Update the batch size

    stage_lowest_cost = float("inf")
    stage_lowest_costs = []
    stage_best_pid = None

    # Create concurrency by allowing multi-processes (even though sim is GPU bound, results were signifactly faster than sharing memory in threading)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(optimize_pid, all_initial_guesses[i], i): i
            for i in range(min(CONC_POPULATION, len(all_initial_guesses)))
        }
        max_population_count = len(futures)
        results = []
        lowest_cost = float("inf")
        lowest_cost_pid = None

        while futures:
            done, futures = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_COMPLETED
            )

            for future in done:
                best_pids, costs, pop_lowest_cost, pop_lowest_cost_pid, epoch = (
                    future.result()
                )
                results.append((pop_lowest_cost, pop_lowest_cost_pid))
                all_pids.append(best_pids)
                all_costs.append(costs)

                if pop_lowest_cost < stage_lowest_cost:
                    stage_lowest_cost = pop_lowest_cost
                    stage_best_pid = pop_lowest_cost_pid
                    stage_lowest_costs.append(stage_lowest_cost)
                    optimal_population_numbers.append(max_population_count)
                    print(
                        f"population #{max_population_count-(CONC_POPULATION-1)} - batch size { BATCH_SIZE } | epoch {epoch}, global minimum cost {stage_lowest_cost}, global optimal PID {stage_best_pid}"
                    )
                else:
                    print(
                        f"population #{max_population_count-(CONC_POPULATION-1)} - batch size { BATCH_SIZE } | epoch {epoch}, no new optimal solution found"
                    )
                if max_population_count < len(all_initial_guesses):
                    futures.add(
                        executor.submit(
                            optimize_pid,
                            all_initial_guesses[max_population_count],
                            max_population_count,
                        )
                    )
                max_population_count += 1

    # select the top results for the next stage, select based on survival rate
    results.sort(key=lambda x: x[0])
    top_guesses = results[: math.ceil(len(results) * top_percentage)]
    all_initial_guesses = [pid for _, pid in top_guesses]

    if stage == len(BATCH_SIZE_STAGES) - 1:
        best_cost, best_pid = top_guesses[0]
        print(f"Optimum PID values: {best_pid}")
        print(f"Sum average total cost: {best_cost}")

        with open("solution.txt", "w") as f:
            f.write(f"Optimum PID values: {best_pid}\n")
            f.write(f"Sum average total cost: {best_cost}\n")

