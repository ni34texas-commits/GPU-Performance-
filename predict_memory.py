import os
import re
import sys
import shutil
import subprocess
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

# ─── CONFIG ────────────────────────────────────────────────────────────────
SRC_BASE = os.path.expanduser("~/GPULLM/ParallelCodeEstimation/src")
PREDICT_LOG = os.path.expanduser("~/GPULLM/benchmarker/predictions_log.txt")

# ─── KNOWLEDGE BASE ────────────────────────────────────────────────────────
# Ground truth labels from real GPU benchmarking on Ada HPC cluster
KNOWN_KERNELS = {
    "stencil1d-cuda": {
        "winner": "DEVICE",
        "device_us": 3.0, "host_us": 5.0, "unified_us": 7.0,
        "pattern": "Sequential stencil sweep, structured grid, high compute intensity, no atomic ops"
    },
    "scan-cuda": {
        "winner": "DEVICE",
        "device_us": 3.4, "host_us": 5.9, "unified_us": 4.9,
        "pattern": "Parallel prefix scan, sequential memory access, shared memory, no atomic ops"
    },
    "backprop-cuda": {
        "winner": "DEVICE",
        "device_us": 114.0, "host_us": 2800.0, "unified_us": 3553.0,
        "pattern": "Neural network matrix ops, large weight matrices, high bandwidth, contiguous access"
    },
    "gaussian-cuda": {
        "winner": "DEVICE",
        "device_us": 408.0, "host_us": 1792.0, "unified_us": 778.0,
        "pattern": "Gaussian elimination, structured grid access, iterative row-based computation"
    },
    "pathfinder-cuda": {
        "winner": "DEVICE",
        "device_us": 34.0, "host_us": 71.0, "unified_us": 521.0,
        "pattern": "Dynamic programming sweep, sequential row access, dependency chain, no atomics"
    },
    "histogram-cuda": {
        "winner": "UNIFIED",
        "device_us": 144.0, "host_us": 1426.0, "unified_us": 135.0,
        "pattern": "Atomic ops on histogram bins, random/scattered pixel access, irregular pattern"
    },
    "kmeans-cuda": {
        "winner": "UNIFIED",
        "device_us": 116.3, "host_us": 202.7, "unified_us": 102.9,
        "pattern": "K-means clustering, atomic updates on cluster centers, random point-to-cluster access"
    },
}

# ─── HELPERS ───────────────────────────────────────────────────────────────
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

def read_source(kernel_name):
    kernel_dir = os.path.join(SRC_BASE, kernel_name)
    if not os.path.exists(kernel_dir):
        return None
    source_file = find_source(kernel_dir)
    if not source_file:
        return None
    with open(source_file, 'r') as f:
        return f.read()[:4000]

def build_knowledge_prompt():
    kb = "=== KNOWLEDGE BASE: PATTERNS LEARNED FROM REAL GPU BENCHMARKS ===\n\n"
    for name, info in KNOWN_KERNELS.items():
        kb += f"Kernel  : {name}\n"
        kb += f"Pattern : {info['pattern']}\n"
        kb += f"Timings : Device={info['device_us']}us | "
        kb += f"Host={info['host_us']}us | Unified={info['unified_us']}us\n"
        kb += f"WINNER  : {info['winner']} memory\n\n"

    kb += "=== DECISION RULES LEARNED ===\n"
    kb += "UNIFIED memory wins when:\n"
    kb += "  - Kernel uses atomic operations on shared counters/centers\n"
    kb += "  - Memory access is random or scattered (not sequential)\n"
    kb += "  - Multiple threads access unpredictable locations\n"
    kb += "  - Examples: histogram-cuda, kmeans-cuda\n\n"
    kb += "DEVICE memory wins when:\n"
    kb += "  - Memory access is sequential and structured\n"
    kb += "  - High compute intensity with coalesced access\n"
    kb += "  - Large data transferred once then computed many times\n"
    kb += "  - Examples: stencil1d, scan, backprop, gaussian, pathfinder\n\n"
    kb += "HOST memory almost always loses due to PCIe overhead.\n\n"
    return kb

def extract_prediction(text):
    # Priority 1: look at final recommendation sentence
    for line in reversed(text.split("\n")):
        line_lower = line.lower()
        if any(w in line_lower for w in ["predict", "recommend", "final", "conclusion", "fastest"]):
            for mem in ["unified", "device", "host"]:
                if mem in line_lower:
                    return mem.upper()
    # Priority 2: last occurrence of memory type
    text_lower = text.lower()
    last_pos = -1
    last_mem = "UNKNOWN"
    for mem in ["unified", "device", "host"]:
        pos = text_lower.rfind(f"{mem} memory")
        if pos > last_pos:
            last_pos = pos
            last_mem = mem.upper()
    return last_mem

# ─── PREDICTION (NO BENCHMARKING) ──────────────────────────────────────────
def predict_memory(kernel_name, source_code):
    llm = ChatOllama(
        model="codellama",
        base_url="http://localhost:11434",
        temperature=0.1
    )

    messages = [
        SystemMessage(content=(
            "You are an expert CUDA GPU performance engineer. "
            "You have benchmarked 7 CUDA kernels on a real GPU and learned "
            "which memory type (Device, Host, Unified) is fastest for each. "
            "Your task is to predict the best memory type for a NEW kernel "
            "by analyzing its source code and comparing it to known patterns. "
            "Key insight: kernels with atomic operations and random/scattered "
            "access prefer UNIFIED memory. Kernels with sequential structured "
            "access prefer DEVICE memory."
        )),
        HumanMessage(content=(
            f"{build_knowledge_prompt()}"
            f"=== NEW KERNEL TO PREDICT ===\n\n"
            f"Kernel Name: {kernel_name}\n\n"
            f"Source Code (first 4000 chars):\n{source_code}\n\n"
            f"=== PREDICTION TASK ===\n\n"
            f"Analyze this kernel and answer:\n\n"
            f"1. Memory access pattern: sequential or random/scattered?\n"
            f"   Does it use atomic operations?\n\n"
            f"2. Which known kernel does it most resemble?\n\n"
            f"3. Predicted fastest memory type and reasoning.\n\n"
            f"4. Confidence: High / Medium / Low\n\n"
            f"IMPORTANT RULES:\n"
            f"- If kernel uses atomics + random access → predict UNIFIED\n"
            f"- If kernel uses sequential structured access → predict DEVICE\n"
            f"- HOST memory is almost never fastest\n\n"
            f"5. Final prediction (one word): DEVICE or UNIFIED or HOST"
        ))
    ]

    response = llm.invoke(messages)
    return response.content

# ─── OPTIONAL BENCHMARKING ──────────────────────────────────────────────────
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

def benchmark_to_verify(kernel_name, binary, args):
    kernel_dir = os.path.join(SRC_BASE, kernel_name)
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
        run_result = subprocess.run(
            f"./{output_bin} {args} 2>&1",
            shell=True, cwd=kernel_dir,
            capture_output=True, text=True, timeout=60
        )
        val = extract_timing(run_result.stdout + run_result.stderr)
        timings[mem_type] = round(val, 3) if val else 9999.0
    return timings

# ─── MAIN PREDICTION RUN ───────────────────────────────────────────────────
def run_prediction(kernel_name, binary="main", args="", verify=False):
    print(f"\n{'='*65}")
    print(f"  GPULLM Memory Predictor")
    print(f"  Kernel : {kernel_name}")
    print(f"  KB size: {len(KNOWN_KERNELS)} kernels")
    print(f"{'='*65}\n")

    # Step 1: Read source
    print("[Step 1] Reading kernel source code...")
    source_code = read_source(kernel_name)
    if not source_code:
        print(f"  ERROR: Cannot find source for '{kernel_name}'")
        print(f"  Check that folder exists in: {SRC_BASE}")
        return
    print(f"  Done ({len(source_code)} chars)\n")

    # Step 2: Show knowledge base
    print(f"[Step 2] Knowledge base ({len(KNOWN_KERNELS)} kernels):")
    for k, info in KNOWN_KERNELS.items():
        tag = " ← THIS KERNEL" if k == kernel_name else ""
        print(f"  {k:<22} → {info['winner']}{tag}")
    print()

    # Step 3: Predict
    print("[Step 3] CodeLlama predicting (no benchmarking)...")
    prediction = predict_memory(kernel_name, source_code)
    predicted_mem = extract_prediction(prediction)

    print(f"\n{'='*65}")
    print(f"CODELLAMA ANALYSIS:")
    print(f"{'='*65}")
    print(prediction)
    print(f"\n{'─'*65}")
    print(f"FINAL PREDICTION: {predicted_mem}")
    print(f"{'='*65}")

    # Step 4: Verify (optional)
    actual_winner = None
    verdict = "NOT VERIFIED"
    if verify:
        print(f"\n[Step 4] Verifying by benchmarking...")
        timings = benchmark_to_verify(kernel_name, binary, args)
        if timings:
            actual_winner = min(timings, key=timings.get).upper()
            verdict = "CORRECT" if predicted_mem == actual_winner else "WRONG"
            print(f"\n  Device  : {timings['device']} us")
            print(f"  Host    : {timings['host']} us")
            print(f"  Unified : {timings['unified']} us")
            print(f"\n  Predicted : {predicted_mem}")
            print(f"  Actual    : {actual_winner}")
            print(f"  Verdict   : {verdict}")

    # Check against known ground truth
    if kernel_name in KNOWN_KERNELS and not verify:
        truth = KNOWN_KERNELS[kernel_name]["winner"]
        verdict = "CORRECT" if predicted_mem == truth else "WRONG"
        print(f"\n  Ground Truth (from KB): {truth}")
        print(f"  Verdict               : {verdict}")

    # Save to log
    with open(PREDICT_LOG, 'a') as f:
        f.write(f"\n{'='*65}\n")
        f.write(f"Kernel      : {kernel_name}\n")
        f.write(f"Predicted   : {predicted_mem}\n")
        if actual_winner:
            f.write(f"Actual      : {actual_winner}\n")
        f.write(f"Verdict     : {verdict}\n")
        f.write(f"Analysis    :\n{prediction}\n")

    print(f"\nLog saved to: {PREDICT_LOG}")

# ─── INTERACTIVE MODE ──────────────────────────────────────────────────────
def interactive_mode():
    print("\n" + "="*65)
    print("  GPULLM Memory Predictor - Interactive Mode")
    print(f"  Knowledge base: {len(KNOWN_KERNELS)} kernels")
    print("  Type 'list' to see all available kernels")
    print("  Type 'exit' to quit")
    print("="*65 + "\n")

    while True:
        kernel_name = input("Kernel name: ").strip()

        if kernel_name.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break

        if kernel_name.lower() == 'list':
            print("\nAvailable kernels:")
            try:
                kernels = sorted([d for d in os.listdir(SRC_BASE)
                                 if os.path.isdir(os.path.join(SRC_BASE, d))
                                 and d.endswith('-cuda')])
                for i, k in enumerate(kernels, 1):
                    tag = " (known)" if k in KNOWN_KERNELS else " (unseen)"
                    print(f"  {i:3}. {k}{tag}")
            except:
                print("  Could not list kernels")
            print()
            continue

        if not kernel_name:
            continue

        verify = input("Verify by benchmarking? (y/n): ").strip().lower() == 'y'
        binary = "main"
        args = ""
        if verify:
            binary = input("Binary name [main]: ").strip() or "main"
            args = input("Arguments [none]: ").strip()

        run_prediction(kernel_name, binary=binary, args=args, verify=verify)
        print()

# ─── ENTRY POINT ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        kernel = sys.argv[1]
        verify = "--verify" in sys.argv
        binary = "main"
        args = ""
        # Parse optional binary and args
        non_flag_args = [a for a in sys.argv[2:] if not a.startswith("--")]
        if len(non_flag_args) > 0:
            binary = non_flag_args[0]
        if len(non_flag_args) > 1:
            args = non_flag_args[1]
        run_prediction(kernel, binary=binary, args=args, verify=verify)
    else:
        interactive_mode()
