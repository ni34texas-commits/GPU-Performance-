import os
import re
import shutil
import subprocess
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

SRC_BASE = os.path.expanduser("~/GPULLM/ParallelCodeEstimation/src")
EVAL_LOG = os.path.expanduser("~/GPULLM/benchmarker/evaluation_report.txt")

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

def benchmark_one(kernel_name):
    binary, args = KERNEL_CONFIGS[kernel_name]
    kernel_dir = os.path.join(SRC_BASE, kernel_name)
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
            timings[mem_type.lower()] = round(sum(vals)/len(vals), 3) if vals else 9999.0
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
    return timings

def get_agent_recommendation(kernel_name, source_code, timings):
    winner = min(timings, key=timings.get)
    llm = ChatOllama(model="codellama", base_url="http://localhost:11434", temperature=0.1)
    messages = [
        SystemMessage(content=(
            "You are an expert CUDA GPU performance engineer. "
            "Based on the kernel source code and benchmark results, "
            "recommend the best memory type and explain why."
        )),
        HumanMessage(content=(
            f"Kernel: {kernel_name}\n\n"
            f"Source Code (first 3000 chars):\n{source_code}\n\n"
            f"Benchmark Results:\n"
            f"  Device Memory  : {timings['device']} microseconds\n"
            f"  Host Memory    : {timings['host']} microseconds\n"
            f"  Unified Memory : {timings['unified']} microseconds\n"
            f"  Measured Winner: {winner.upper()} memory\n\n"
            f"1. Which memory type is fastest?\n"
            f"2. Why does it perform best?\n"
            f"3. Final recommendation: Device, Host, or Unified memory?"
        ))
    ]
    response = llm.invoke(messages)
    return response.content, winner

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

def run_evaluation():
    print("\n" + "="*70)
    print("  STEP 5: AGENT ACCURACY EVALUATION")
    print("  Comparing Agent Recommendations vs Ground Truth Labels")
    print("="*70)

    results = []

    for kernel_name in KERNEL_CONFIGS:
        print(f"\n[Evaluating] {kernel_name}")
        print("-"*50)

        truth = GROUND_TRUTH[kernel_name]
        kernel_dir = os.path.join(SRC_BASE, kernel_name)
        source_file = find_source(kernel_dir)
        with open(source_file, 'r') as f:
            source_code = f.read()[:3000]

        print(f"  Benchmarking...", end=" ", flush=True)
        timings = benchmark_one(kernel_name)
        measured_winner = min(timings, key=timings.get)
        print(f"done. Measured: {measured_winner.upper()}")

        print(f"  Asking CodeLlama...", end=" ", flush=True)
        agent_answer, _ = get_agent_recommendation(kernel_name, source_code, timings)
        agent_rec = extract_recommendation(agent_answer)
        print(f"done. Agent says: {agent_rec.upper()}")

        verdict = "CORRECT" if agent_rec == truth else "WRONG"
        speedup = round(timings['host'] / timings[measured_winner], 1)

        print(f"  Ground Truth : {truth.upper()}")
        print(f"  Agent Rec    : {agent_rec.upper()}")
        print(f"  Verdict      : {verdict}")

        results.append({
            'kernel': kernel_name, 'truth': truth,
            'measured': measured_winner, 'agent_rec': agent_rec,
            'verdict': verdict, 'device_us': timings['device'],
            'host_us': timings['host'], 'unified_us': timings['unified'],
            'speedup': speedup, 'agent_answer': agent_answer,
        })

    correct = sum(1 for r in results if r['verdict'] == 'CORRECT')
    total = len(results)
    accuracy = correct / total * 100

    print("\n\n" + "="*70)
    print("  EVALUATION REPORT")
    print("="*70)
    print(f"\n{'Kernel':<20} {'Truth':<10} {'Agent':<10} {'Measured':<10} {'Verdict':<10} {'Speedup'}")
    print("-"*70)
    for r in results:
        icon = "✓" if r['verdict'] == 'CORRECT' else "✗"
        print(f"{r['kernel']:<20} {r['truth'].upper():<10} {r['agent_rec'].upper():<10} "
              f"{r['measured'].upper():<10} {icon} {r['verdict']:<8} {r['speedup']}x")
    print("-"*70)
    print(f"\nOverall Accuracy : {correct}/{total} = {accuracy:.1f}%")
    print(f"Correct          : {correct} kernels")
    print(f"Wrong            : {total - correct} kernels")

    print(f"\nGround Truth Distribution:")
    for mem in ['device', 'host', 'unified']:
        count = sum(1 for r in results if r['truth'] == mem)
        print(f"  {mem.upper():<10}: {count} kernels")

    print(f"\nAgent Recommendation Distribution:")
    for mem in ['device', 'host', 'unified', 'unknown']:
        count = sum(1 for r in results if r['agent_rec'] == mem)
        if count > 0:
            print(f"  {mem.upper():<10}: {count} kernels")

    with open(EVAL_LOG, 'w') as f:
        f.write("STEP 5: EVALUATION REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"{'Kernel':<20} {'Truth':<10} {'Agent':<10} {'Measured':<10} {'Verdict'}\n")
        f.write("-"*70 + "\n")
        for r in results:
            f.write(f"{r['kernel']:<20} {r['truth'].upper():<10} {r['agent_rec'].upper():<10} "
                    f"{r['measured'].upper():<10} {r['verdict']}\n")
        f.write("-"*70 + "\n")
        f.write(f"\nOverall Accuracy: {correct}/{total} = {accuracy:.1f}%\n\n")
        for r in results:
            f.write(f"\n{'='*50}\n")
            f.write(f"Kernel    : {r['kernel']}\n")
            f.write(f"Truth     : {r['truth'].upper()}\n")
            f.write(f"Agent Rec : {r['agent_rec'].upper()}\n")
            f.write(f"Verdict   : {r['verdict']}\n")
            f.write(f"Timings   : Device={r['device_us']}us | Host={r['host_us']}us | Unified={r['unified_us']}us\n")
            f.write(f"Speedup   : {r['speedup']}x vs host\n")
            f.write(f"Agent Answer:\n{r['agent_answer']}\n")

    print(f"\nFull report saved to: {EVAL_LOG}")
    print("="*70)
    return accuracy

if __name__ == "__main__":
    run_evaluation()
