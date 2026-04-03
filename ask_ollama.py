import sys
import requests
import json
import os

# Get arguments from shell script
kernel_name  = sys.argv[1]
device_time  = sys.argv[2]
host_time    = sys.argv[3]
unified_time = sys.argv[4]

# Build prompt for Ollama
prompt = f"""
You are a GPU performance expert.

A CUDA kernel called '{kernel_name}' was benchmarked with 3 memory types:
- Device memory  : {device_time} microseconds
- Host memory    : {host_time} microseconds  
- Unified memory : {unified_time} microseconds

Based on these timings:
1. Which memory type is the fastest?
2. Why do you think it performed best?
3. What is your recommendation for this kernel?

Give a short, clear answer.
"""

print("\n====================================")
print("Asking Ollama (llama3)...")
print("====================================")

# Call Ollama API
response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }
)

# Parse and print response
result = response.json()
answer = result["response"]

print(f"\nOllama Recommendation for '{kernel_name}':")
print("------------------------------------")
print(answer)
print("------------------------------------")

# Save recommendation to file
#with open("/root/GPULLM/benchmarker/recommendations.txt", "a") as f:
with open(os.path.expanduser("~/GPULLM/benchmarker/recommendations.txt"), "a") as f:
    f.write(f"\nKernel: {kernel_name}\n")
    f.write(f"Device: {device_time}us | Host: {host_time}us | Unified: {unified_time}us\n")
    f.write(f"Ollama says: {answer}\n")
    f.write("-" * 50 + "\n")

print("\nRecommendation saved to recommendations.txt")

