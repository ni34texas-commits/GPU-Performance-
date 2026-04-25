import os
import re
import sys
import shutil
import subprocess
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

# ─── CONFIG ────────────────────────────────────────────────────────────────
SRC_BASE = os.path.expanduser("~/GPULLM/ParallelCodeEstimation/src")
LOG_FILE  = os.path.expanduser("~/GPULLM/benchmarker/recommendations.txt")

# ─── GROUND TRUTH ──────────────────────────────────────────────────────────
# Real GPU benchmark results on Ada HPC cluster (sm_60)
GROUND_TRUTH = {
    "stencil1d-cuda":  "device",
    "scan-cuda":       "device",
    "backprop-cuda":   "device",
    "gaussian-cuda":   "device",
    "pathfinder-cuda": "device",
    "histogram-cuda":  "unified",
    "kmeans-cuda":     "unified",
}

# ─── KERNEL CONFIGS ────────────────────────────────────────────────────────
# (binary_name, run_arguments)
KERNEL_CONFIGS = {
    "stencil1d-cuda":  ("stencil_1d", "1024 100"),
    "scan-cuda":       ("main",       "1024 100"),
    "backprop-cuda":   ("main",       "65536"),
    "gaussian-cuda":   ("main",       "-s 64 -t"),
    "pathfinder-cuda": ("main",       "1000 10 5"),
    "histogram-cuda":  ("main",       None),
    "kmeans-cuda":     ("kmeans",     "-i /tmp/kmeans_input.txt -m 5 -n 5"),
}

# ─── PRE-MEASURED TIMINGS ──────────────────────────────────────────────────
# Kernels that cannot be auto-modified use pre-measured real GPU timings
PRE_MEASURED = {
    "kmeans-cuda": {
        "device":  116.3,
        "host":    202.7,
        "unified": 102.9,
    }
}

# ─── SOURCE FINDER ─────────────────────────────────────────────────────────
def find_source(kernel_dir):
    for fname in ["main.cu", "stencil_1d.cu", "gaussianElim.cu",
                  "backprop.cu", "cluster.cu"]:
        fpath = os.path.join(kernel_dir, fname)
        if os.path.exists(fpath):
            return fpath
    for f in os.listdir(kernel_dir):
        if f.endswith(".cu"):
            return os.path.join(kernel_dir, f)
    return None

# ─── TIMING EXTRACTOR ──────────────────────────────────────────────────────
def extract_timing(output):
    patterns = [
        (r'Kernel time\s*=\s*(\d+\.?\d+)\s*\(us\)', 1.0),
        (r'(?:Total k|K)ernel(?: execution)? time\D+(\d+\.?\d*)\s*\(us\)', 1.0),
        (r'(?:Total k|K)ernel(?: execution)? time[^\n=]*[=: ]+(\d+\.?\d*)\s*\(s\)', 1e6),
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

# ─── MEMORY VERSION CREATOR ────────────────────────────────────────────────
def create_memory_versions(source_file):
    with open(source_file, 'r') as f:
        original = f.read()
    host_code = re.sub(r'cudaMalloc\s*\(\s*\(void\s*\*\*\)\s*&',
                       'cudaMallocHost((void **)&', original)
    host_code = re.sub(r'cudaFree\s*\(', 'cudaFreeHost(', host_code)
    host_code = re.sub(r'cudaMemcpy\s*\([^;]+cudaMemcpyHostToDevice[^;]+;',
                       '// removed', host_code)
    host_code = re.sub(r'cudaMemcpy\s*\([^;]+cudaMemcpyDeviceToHost[^;]+;',
                       '// removed', host_code)
    unified_code = re.sub(r'cudaMalloc\s*\(\s*\(void\s*\*\*\)\s*&',
                          'cudaMallocManaged((void **)&', original)
    unified_code = re.sub(r'cudaMemcpy\s*\([^;]+cudaMemcpyHostToDevice[^;]+;',
                          '// removed', unified_code)
    unified_code = re.sub(r'cudaMemcpy\s*\([^;]+cudaMemcpyDeviceToHost[^;]+;',
                          '// removed', unified_code)
    return {'device': original, 'host': host_code, 'unified': unified_code}

# ─── BENCHMARKER ───────────────────────────────────────────────────────────
def benchmark_kernel(kernel_name):
    binary, args = KERNEL_CONFIGS[kernel_name]
    kernel_dir = os.path.join(SRC_BASE, kernel_name)

    # Use pre-measured timings for kernels that cannot be auto-modified
    if kernel_name in PRE_MEASURED:
        timings = PRE_MEASURED[kernel_name]
        fastest = min(timings, key=timings.get)
        speedup = round(timings['host'] / timings[fastest], 1)
        return {
            "device":  timings['device'],
            "host":    timings['host'],
            "unified": timings['unified'],
            "fastest": fastest,
            "speedup": speedup,
        }

    # Histogram has built-in 3-way memory test
    if kernel_name == "histogram-cuda":
        result = subprocess.run("./main 2>&1", shell=True, cwd=kernel_dir,
                                capture_output=True, text=True, timeout=60)
        output = result.stdout + result.stderr
        timings = {}
        for mem_type in ["DEVICE", "HOST", "UNIFIED"]:
            sections = output.split(f'Shared memory atomics {mem_type} \tPASS')
            vals = []
            for section in sections[1:]:
                next_sec = re.split(r'Shared memory atomics (?:DEVICE|HOST|UNIFIED)', section)
                avg_matches = re.findall(r'Avg time (\d+\.?\d+) us', next_sec[0])
                vals.extend([float(x) for x in avg_matches])
            timings[mem_type.lower()] = round(sum(vals)/len(vals), 3) if vals else 9999.0
        fastest = min(timings, key=timings.get)
        speedup = round(timings['host'] / timings[fastest], 1)
        return {**timings, "fastest": fastest, "speedup": speedup}

    # Standard auto-modification benchmark
    source_file = find_source(kernel_dir)
    if not source_file:
        return None
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
        run_args = args if args else ""
        run_result = subprocess.run(
            f"./{output_bin} {run_args} 2>&1",
            shell=True, cwd=kernel_dir,
            capture_output=True, text=True, timeout=60
        )
        val = extract_timing(run_result.stdout + run_result.stderr)
        timings[mem_type] = round(val, 3) if val else 9999.0

    fastest = min(timings, key=timings.get)
    speedup = round(timings['host'] / timings[fastest], 1)
    return {**timings, "fastest": fastest, "speedup": speedup}

# ─── CODELLAMA ANALYZER ────────────────────────────────────────────────────
def analyze_with_codellama(kernel_name, source_code, timings):
    llm = ChatOllama(
        model="codellama",
        base_url="http://localhost:11434",
        temperature=0.1
    )

    fastest = timings['fastest'].upper()

    messages = [
        SystemMessage(content=(
            "You are an expert CUDA GPU performance engineer. "
            "Based on the kernel source code and benchmark timing results, "
            "recommend the best memory type (Device, Host, or Unified) and explain why. "
            "Always end with a clear final recommendation."
        )),
        HumanMessage(content=(
            f"Kernel: {kernel_name}\n\n"
            f"Source Code (first 3000 chars):\n{source_code}\n\n"
            f"Benchmark Results:\n"
            f"  Device Memory  : {timings['device']} microseconds\n"
            f"  Host Memory    : {timings['host']} microseconds\n"
            f"  Unified Memory : {timings['unified']} microseconds\n"
            f"  Measured Winner: {fastest} memory\n\n"
            f"1. Which memory type is fastest based on the benchmarks?\n"
            f"2. Why does this memory type perform best for this kernel?\n"
            f"3. Final recommendation: Device, Host, or Unified memory?"
        ))
    ]

    response = llm.invoke(messages)
    return response.content

# ─── RECOMMENDATION EXTRACTOR ──────────────────────────────────────────────
def extract_recommendation(agent_answer):
    # Priority 1: look at final recommendation sentence
    for line in reversed(agent_answer.split("\n")):
        line_lower = line.lower()
        if any(w in line_lower for w in ["recommend", "use", "final", "conclusion"]):
            for mem in ["device", "unified", "host"]:
                if mem in line_lower:
                    return mem
    # Priority 2: last occurrence of memory type
    answer_lower = agent_answer.lower()
    last_pos = -1
    last_mem = "unknown"
    for mem in ["device", "unified", "host"]:
        pos = answer_lower.rfind(f"{mem} memory")
        if pos > last_pos:
            last_pos = pos
            last_mem = mem
    return last_mem

# ─── MAIN AGENT RUN ────────────────────────────────────────────────────────
def run_agent(kernel_name):
    print(f"\n{'='*60}")
    print(f"  LangChain + CodeLlama CUDA Memory Agent")
    print(f"  Kernel: {kernel_name}")
    print(f"{'='*60}\n")

    kernel_dir = os.path.join(SRC_BASE, kernel_name)
    source_file = find_source(kernel_dir)
    if not source_file:
        print(f"ERROR: Source not found for {kernel_name}")
        return

    # Step 1: Read source code
    print("[Step 1] Reading source code...")
    with open(source_file, 'r') as f:
        source_code = f.read()[:3000]
    print(f"  Done ({len(source_code)} chars)\n")

    # Step 2: Benchmark
    print("[Step 2] Benchmarking all 3 memory types...")
    if kernel_name in PRE_MEASURED:
        print("  (Using pre-measured real GPU timings for this kernel)")
    timings = benchmark_kernel(kernel_name)
    if not timings:
        print("  ERROR: Benchmarking failed")
        return
    print(f"  Device Memory  : {timings['device']} us")
    print(f"  Host Memory    : {timings['host']} us")
    print(f"  Unified Memory : {timings['unified']} us")
    print(f"  Fastest        : {timings['fastest'].upper()} memory")
    print(f"  Speedup vs Host: {timings['speedup']}x\n")

    # Step 3: Ask CodeLlama
    print("[Step 3] Asking CodeLlama to analyze results...")
    answer = analyze_with_codellama(kernel_name, source_code, timings)
    agent_rec = extract_recommendation(answer)

    # Step 4: Compare vs ground truth
    truth = GROUND_TRUTH.get(kernel_name, "unknown")
    verdict = "CORRECT" if agent_rec == truth else "WRONG"

    print(f"\n{'='*60}")
    print(f"CODELLAMA RECOMMENDATION:")
    print(f"{'='*60}")
    print(answer)
    print(f"\nGround Truth : {truth.upper()}")
    print(f"Agent Rec    : {agent_rec.upper()}")
    print(f"Verdict      : {verdict}")
    print(f"{'='*60}")

    # Save to log
    with open(LOG_FILE, 'a') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Kernel    : {kernel_name}\n")
        f.write(f"Truth     : {truth.upper()}\n")
        f.write(f"Agent Rec : {agent_rec.upper()}\n")
        f.write(f"Verdict   : {verdict}\n")
        f.write(f"Timings   : Device={timings['device']}us | "
                f"Host={timings['host']}us | Unified={timings['unified']}us\n")
        f.write(f"Answer    :\n{answer}\n")

    return agent_rec, verdict

# ─── INTERACTIVE MODE ──────────────────────────────────────────────────────
def interactive_mode():
    print("\n" + "="*60)
    print("  GPULLM Interactive Agent")
    print("  Type a kernel name to analyze, ask CUDA questions,")
    print("  or type 'exit' to quit")
    print("="*60 + "\n")

    llm = ChatOllama(
        model="codellama",
        base_url="http://localhost:11434",
        temperature=0.1
    )

    print("Known kernels:")
    for k in KERNEL_CONFIGS:
        truth = GROUND_TRUTH.get(k, "unknown")
        print(f"  {k:<22} → ground truth: {truth.upper()}")
    print()

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break
        if not user_input:
            continue

        # If it's a kernel name — run full agent
        if user_input in KERNEL_CONFIGS:
            run_agent(user_input)
        else:
            # Otherwise ask CodeLlama as a CUDA question
            response = llm.invoke([
                SystemMessage(content="You are an expert CUDA GPU performance engineer."),
                HumanMessage(content=user_input)
            ])
            print(f"\nCodeLlama: {response.content}\n")

# ─── ENTRY POINT ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    if "--interactive" in sys.argv:
        interactive_mode()
    elif len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        kernel = sys.argv[1]
        if kernel not in KERNEL_CONFIGS:
            print(f"ERROR: Unknown kernel '{kernel}'")
            print(f"Available: {list(KERNEL_CONFIGS.keys())}")
            sys.exit(1)
        run_agent(kernel)
    else:
        # Run all kernels
        print("\n" + "="*60)
        print("  Running agent on ALL kernels")
        print("="*60)
        results = []
        for kernel_name in KERNEL_CONFIGS:
            result = run_agent(kernel_name)
            if result:
                results.append((kernel_name, result[0], result[1]))

        print(f"\n{'='*60}")
        print("  FINAL SUMMARY")
        print(f"{'='*60}")
        correct = sum(1 for _, _, v in results if v == "CORRECT")
        print(f"\n{'Kernel':<22} {'Agent':<10} {'Verdict'}")
        print("-"*45)
        for kernel, rec, verdict in results:
            icon = "✓" if verdict == "CORRECT" else "✗"
            print(f"{kernel:<22} {rec.upper():<10} {icon} {verdict}")
        print("-"*45)
        print(f"Accuracy: {correct}/{len(results)} = {correct/len(results)*100:.1f}%")
