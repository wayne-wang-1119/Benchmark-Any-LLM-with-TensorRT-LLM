import subprocess
import concurrent.futures
import numpy as np
import pandas as pd
import time


# Adapt this function to call your TensorRT-LLM benchmark script and return latency
def benchmark_tensorrt_llm(input_length, output_length):
    start_time = time.time()
    command = f"python3 benchmark.py -m mixtral_7b --mode plugin --batch_size 1 --input_output_len {input_length},{output_length}"
    subprocess.run(command.split(), capture_output=True)
    end_time = time.time()
    return end_time - start_time


def worker_task(_):
    latencies = []
    for _ in range(32):  # Each worker performs 32 calls
        prompt_length = int(np.random.normal(loc=50, scale=15))
        input_length = max(5, min(prompt_length, 100))
        max_tokens = int(np.random.normal(loc=200, scale=50))
        output_length = max(50, min(max_tokens, 512))
        latency = benchmark_tensorrt_llm(input_length, output_length)
        latencies.append(latency)
    return latencies


def run_experiment(num_workers):
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(worker_task, range(num_workers)))
    # Flatten results and calculate metrics
    all_latencies = [
        latency for worker_latencies in results for latency in worker_latencies
    ]
    average_latency = np.mean(all_latencies)
    throughput = (32 * num_workers) / sum(all_latencies)
    return throughput, average_latency


worker_counts = [4, 6, 8, 12, 16, 20, 32]
results = []

for num_workers in worker_counts:
    throughput, average_latency = run_experiment(num_workers)
    results.append(
        {
            "Workers": num_workers,
            "Throughput": throughput,
            "Average Latency": average_latency,
        }
    )

# Save to CSV
df = pd.DataFrame(results)
df.to_csv("tensorrt_llm_results.csv", index=False)
