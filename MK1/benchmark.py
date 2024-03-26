import concurrent.futures
import time
import numpy as np
import pandas as pd
import requests
import json


def generate_prompt_length() -> int:
    """Generate a prompt length based on a normal distribution."""
    prompt_length = int(np.random.normal(loc=50, scale=15))
    return max(5, min(prompt_length, 100))  # Ensure length is within a reasonable range


def generate_max_tokens() -> int:
    """Generate a max_tokens value based on a normal distribution."""
    max_tokens = int(np.random.normal(loc=200, scale=50))
    return max(50, min(max_tokens, 512))  # Ensure tokens are within a reasonable range


def make_llm_call(prompt: str, max_tokens: int) -> float:
    """Make a call to the LLM endpoint and return the latency."""
    url = "https://mk1--mk1-chat-endpoint-dev.modal.run/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "text": prompt,
        "max_tokens": max_tokens,
        "eos_token_ids": [1, 2],
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 1.0,
    }
    start_time = time.time()
    response = requests.post(url, headers=headers, data=json.dumps(data))
    end_time = time.time()
    return end_time - start_time


def worker_task():
    """The task each worker will perform."""
    latencies = []
    for _ in range(32):  # 32 sequential calls
        prompt_length = generate_prompt_length()
        prompt = "Sample text " * prompt_length
        max_tokens = generate_max_tokens()
        latency = make_llm_call(prompt, max_tokens)
        latencies.append(latency)
    return latencies


def run_experiment(num_workers: int):
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker_task) for _ in range(num_workers)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    # Flatten the list of latencies and calculate metrics
    all_latencies = [
        latency for worker_latencies in results for latency in worker_latencies
    ]
    average_latency = sum(all_latencies) / len(all_latencies)
    total_time = max(
        sum(worker_latencies) for worker_latencies in results
    )  # Maximum total time across all workers
    throughput = 32 * num_workers / total_time

    return throughput, average_latency


# List of worker counts to test
worker_counts = [4, 6, 8, 12, 16, 20, 32]
results = []

for num_workers in worker_counts:
    throughput, average_latency = run_experiment(num_workers)
    print(
        f"Workers: {num_workers}, Throughput: {throughput}, Average Latency: {average_latency}"
    )
    results.append(
        {
            "Workers": num_workers,
            "Throughput": throughput,
            "Average Latency": average_latency,
        }
    )

# Save results to a CSV file
df = pd.DataFrame(results)
df.to_csv("mk1.csv", index=False)
