import os
import re
import sys
import shutil
import subprocess
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

# ─── CONFIG ────────────────────────────────────────────────────────────────
SRC_BASE = os.path.expanduser("~/GPULLM/ParallelCodeEstimation/src")
AGENT_LOG = os.path.expanduser("~/GPULLM/benchmarker/langchain_agent_log.txt")

GROUND_TRUTH = {
    "stencil1d-cuda":  "device",
    "scan-cuda":       "device",
    "backprop-cuda":   "device",
    "gaussian-cuda":   "device",
    "pathfinder-cuda": "device",
    "histogram-cuda":  "unified",
}

KERNEL_CONFIGS = {
    "stencil1d-cuda":  ("stencil_1d", "1024 100"),
    "scan-cuda":       ("main",       "1024 100"),
    "backprop-cuda":   ("main",       "65536"),
    "gaussian-cuda":   ("main",       "-s 64 -t"),
    "pathfinder-cuda": ("main",       "1000 10 5"),
    "histogram-cuda":  ("main",       None),
}

# ─── HELPERS ───────────────────────────────────────────────────────────────
def find_source(kernel_dir):
    for fname in ["main.cu", "stencil_1d.cu", "gaussianElim.cu", "backprop.cu"]:
        fpath = os.path.join(kernel_dir, fname)
        if os.path.exists(fpath):
            return fpath
    for f in os.listdir(kernel_dir):
        if f.endswith(".cu"):
            return os.path.join(kernel_dir, f)
    return None

def extract_timing(output):
    patterns = [
        (r'(?:Total k|K)ernel(?: execution)? time\D+(\d+\.?\d*)\s*\(us\)', 1.0),
        (r'(?:Total k|K)ernel(?: execution)? time[^\n=]*[=: ]+(\d+\.?\d*)\s*\(s\)', 1e6),
        (r'Kernel time\s*=\s*(\d+\.?\d+)\s*\(us\)', 1.0),
        (r'Average kernel execution time:\s*(\d+\.?\d+)\s*\(s\)', 1e6),
        (r'Average execution time[^:]*:\s*(\d+\.?\d+)\s*\(us\)', 1.0),
        (r'[Kk]ernel time[^:]*:\s*(\d+\.?\d+)\s*ms', 1000.0),
        (r'(\d+\.?\d+)\s*\(us\)', 1.0),
    ]
    for pattern, multiplier in patterns:
        matches = re.findall(pattern, output)
        if matches:
            return float(matches[0]) * multiplier
    return None

def create_memory_versions(source_file):
    with open(source_file, 'r') as f:
        original = f.read()
    host_code = re.sub(r'cudaMalloc\s*\(\s*\(void\s*\*\*\)\s*&', 'cudaMallocHost((void **)&', original)
    host_code = re.sub(r'cudaFree\s*\(', 'cudaFreeHost(', host_code)
    host_code = re.sub(r'cudaMemcpy\s*\([^;]+cudaMemcpyHostToDevice[^;]+;', '// removed', host_code)
    host_code = re.sub(r'cudaMemcpy\s*\([^;]+cudaMemcpyDeviceToHost[^;]+;', '// removed', host_code)
    unified_code = re.sub(r'cudaMalloc\s*\(\s*\(void\s*\*\*\)\s*&', 'cudaMallocManaged((void **)&', original)
    unified_code = re.sub(r'cudaMemcpy\s*\([^;]+cudaMemcpyHostToDevice[^;]+;', '// removed', unified_code)
    unified_code = re.sub(r'cudaMemcpy\s*\([^;]+cudaMemcpyDeviceToHost[^;]+;', '// removed', unified_code)
    return {'device': original, 'host': host_code, 'unified': unified_code}

# ─── TOOL 1: Read Kernel Source Code ───────────────────────────────────────
@tool
def read_kernel_source(kernel_name: str) -> str:
    """Read the CUDA source code of a kernel. Input: kernel folder name e.g. backprop-cuda"""
    kernel_name = kernel_name.strip()
    kernel_dir = os.path.join(SRC_BASE, kernel_name)
    source_file = find_source(kernel_dir)
    if not source_file:
        return f"ERROR: No source found for {kernel_name}"
    with open(source_file, 'r') as f:
        code = f.read()[:3000]
    return f"CUDA Source Code of '{kernel_name}':\n\n{code}"

# ─── TOOL 2: Benchmark Kernel ───────────────────────────────────────────────
@tool
def benchmark_kernel(kernel_name: str) -> str:
    """Benchmark a CUDA kernel with Device, Host, and Unified memory using nvcc and cudaEvent_t timers."""
    kernel_name = kernel_name.strip()
    if kernel_name not in KERNEL_CONFIGS:
        return f"ERROR: '{kernel_name}' not found. Available: {list(KERNEL_CONFIGS.keys())}"

    binary, args = KERNEL_CONFIGS[kernel_name]
    kernel_dir = os.path.join(SRC_BASE, kernel_name)

    # Special case: histogram has built-in memory tests
    if kernel_name == "histogram-cuda":
        result = subprocess.run("./main 2>&1", shell=True, cwd=kernel_dir,
                                capture_output=True, text=True, timeout=60)
        output = result.stdout + result.stderr
        timings = {}
        for mem_type in ["DEVICE", "HOST", "UNIFIED"]:
            sections = output.split(f'Shared memory atomics {mem_type} \tPASS')
            vals = []
            for section in sections[1:]:
                next_section = re.split(r'Shared memory atomics (?:DEVICE|HOST|UNIFIED)', section)
                avg_matches = re.findall(r'Avg time (\d+\.?\d+) us', next_section[0])
                vals.extend([float(x) for x in avg_matches])
            timings[mem_type.lower()] = round(sum(vals) / len(vals), 3) if vals else 9999.0
    else:
        source_file = find_source(kernel_dir)
        versions = create_memory_versions(source_file)
        timings = {}
        for mem_type, modified_code in versions.items():
            backup = source_file + ".backup"
            shutil.copy(source_file, backup)
            with open(source_file, 'w') as f:
                f.write(modified_code)
            output_bin = f"{binary}_{mem_type}"
            compile_result = subprocess.run(
                f"make 2>&1 && mv {binary} {output_bin}",
                shell=True, cwd=kernel_dir,
                capture_output=True, text=True, timeout=60
            )
            shutil.copy(backup, source_file)
            if compile_result.returncode != 0:
                timings[mem_type] = 9999.0
                continue
            run_result = subprocess.run(
                f"./{output_bin} {args} 2>&1",
                shell=True, cwd=kernel_dir,
                capture_output=True, text=True, timeout=60
            )
            val = extract_timing(run_result.stdout + run_result.stderr)
            timings[mem_type] = round(val, 3) if val else 9999.0

    winner = min(timings, key=timings.get)
    speedup = round(timings['host'] / timings[winner], 1)
    return (
        f"Benchmark Results for '{kernel_name}':\n"
        f"  Device Memory  : {timings.get('device')} us\n"
        f"  Host Memory    : {timings.get('host')} us\n"
        f"  Unified Memory : {timings.get('unified')} us\n"
        f"  Fastest        : {winner.upper()} memory\n"
        f"  Speedup vs Host: {speedup}x"
    )

# ─── AGENT ANALYSIS ────────────────────────────────────────────────────────
def analyze_with_codellama(kernel_name, source_result, benchmark_result):
    llm = ChatOllama(model="codellama", base_url="http://localhost:11434", temperature=0.1)
    messages = [
        SystemMessage(content=(
            "You are an expert CUDA GPU performance engineer. "
            "Analyze the kernel source code and benchmark results. "
            "Give a clear recommendation on which memory type to use and why."
        )),
        HumanMessage(content=(
            f"Kernel: {kernel_name}\n\n"
            f"{source_result}\n\n"
            f"{benchmark_result}\n\n"
            f"Based on BOTH the source code memory access patterns AND benchmark results:\n"
            f"1. Which memory type is fastest?\n"
            f"2. Why does it perform best?\n"
            f"3. Final recommendation: Device, Host, or Unified memory?"
        ))
    ]
    response = llm.invoke(messages)
    return response.content

# ─── RUN AGENT ─────────────────────────────────────────────────────────────
def run_agent(kernel_name):
    print(f"\n{'='*60}")
    print(f"  LangChain + CodeLlama CUDA Memory Agent")
    print(f"  Kernel: {kernel_name}")
    print(f"{'='*60}\n")

    # Step 1: Read source code
    print("[Step 1] Reading source code...")
    source_result = read_kernel_source.invoke(kernel_name)
    print(f"  Done ({len(source_result)} chars)\n")

    # Step 2: Benchmark all 3 memory types
    print("[Step 2] Benchmarking all 3 memory types...")
    benchmark_result = benchmark_kernel.invoke(kernel_name)
    print(f"  {benchmark_result}\n")

    # Step 3: Ask CodeLlama to analyze
    print("[Step 3] Asking CodeLlama to analyze results...")
    answer = analyze_with_codellama(kernel_name, source_result, benchmark_result)

    # Evaluate against ground truth
    truth = GROUND_TRUTH.get(kernel_name, "unknown")
    verdict = "CORRECT" if truth in answer.lower() else "WRONG"

    print(f"\n{'='*60}")
    print(f"CODELLAMA RECOMMENDATION:")
    print(f"{'='*60}")
    print(answer)
    print(f"\nGround Truth : {truth.upper()}")
    print(f"Verdict      : {verdict}")
    print(f"{'='*60}")

    # Save to log
    with open(AGENT_LOG, 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Kernel       : {kernel_name}\n")
        f.write(f"Ground Truth : {truth.upper()}\n")
        f.write(f"Verdict      : {verdict}\n")
        f.write(f"Benchmark    :\n{benchmark_result}\n")
        f.write(f"Agent Answer :\n{answer}\n")

    return answer, verdict

# ─── INTERACTIVE MODE ──────────────────────────────────────────────────────
def interactive_mode():
    llm = ChatOllama(model="codellama", base_url="http://localhost:11434", temperature=0.1)
    history = []

    print("\n" + "="*60)
    print("  CUDA Memory Agent - Interactive Mode")
    print("  Type a kernel name to benchmark it:")
    print("  " + ", ".join(KERNEL_CONFIGS.keys()))
    print("  Or ask any CUDA question.")
    print("  Type 'exit' to quit.")
    print("="*60 + "\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        if not user_input:
            continue

        # If user typed a kernel name, run full analysis
        if user_input in KERNEL_CONFIGS:
            print(f"\nDetected kernel name. Running full analysis...\n")
            run_agent(user_input)
            continue

        # Otherwise treat as a general CUDA question
        history.append(HumanMessage(content=user_input))
        messages = [
            SystemMessage(content=(
                "You are an expert CUDA GPU performance engineer. "
                "Help developers understand CUDA memory types (Device, Host, Unified), "
                "kernel optimization, and GPU performance. "
                "Available kernels: " + str(list(KERNEL_CONFIGS.keys()))
            ))
        ] + history

        response = llm.invoke(messages)
        answer = response.content
        history.append(response)
        print(f"\nAgent: {answer}\n")

# ─── MAIN ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] in ["--interactive", "-i"]:
            interactive_mode()
        else:
            run_agent(sys.argv[1])
    else:
        # Run all kernels and show accuracy
        correct = 0
        total = 0
        for kernel in KERNEL_CONFIGS:
            _, verdict = run_agent(kernel)
            if verdict == "CORRECT":
                correct += 1
            total += 1
        print(f"\n{'='*60}")
        print(f"OVERALL ACCURACY: {correct}/{total} = {correct/total*100:.1f}%")
        print(f"{'='*60}")
