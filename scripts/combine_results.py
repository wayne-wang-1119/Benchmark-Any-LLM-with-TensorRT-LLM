import pandas as pd
import matplotlib.pyplot as plt

# Load CSV files
df1 = pd.read_csv("mk1.csv")
df2 = pd.read_csv("tensorrt_llm_results.csv")

plt.figure(figsize=(10, 6))
plt.plot(df1["Average Latency"], df1["Throughput"], marker="o", label="Mk1")
plt.plot(df2["Average Latency"], df2["Throughput"], marker="o", label="TensorRT-LLM")
plt.xlabel("Average Latency (s)")
plt.ylabel("Throughput")
plt.title("Throughput vs. Average Latency")
plt.legend()
plt.grid(True)
plt.savefig("/usr/src/app/throughput_vs_latency.png")
plt.close()
