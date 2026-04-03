import os
import re
import shutil
import subprocess

SRC_BASE = os.path.expanduser("~/GPULLM/ParallelCodeEstimation/src")
RESULTS_CSV = os.path.expanduser("~/GPULLM/benchmarker/results.csv")
RECOMMENDATIONS = os.path.expanduser("~/GPULLM/benchmarker/recommendations.txt")

KERNELS = {
    "stencil1d-cuda":  ("stencil_1d", "1024 100"),
    "scan-cuda":       ("main",       "1024 100"),
    "backprop-cuda":   ("main",       "65536"),
    "gaussian-cuda":   ("main",       "-s 64 -t"),
    "pathfinder-cuda": ("main",       "1000 10 5"),
}

def find_main_cu(kernel_dir):
    for fname in ["main.cu", "stencil_1d.cu", "gaussianElim.cu", "backprop.cu"]:
        fpath = os.path.join(kernel_dir, fname)
        if os.path.exists(fpath):
            return fpath
    for f in os.listdir(kernel_dir):
        if f.endswith(".cu"):
            return os.path.join(kernel_dir, f)
    return None

def create_memory_versions(source_file):
    with open(source_file, 'r') as f:
        original = f.read()

    # HOST version
    host_code = re.sub(
        r'cudaMalloc\s*\(\s*\(void\s*\*\*\)\s*&',
        'cudaMallocHost((void **)&',
        original
    )
    host_code = re.sub(r'cudaFree\s*\(', 'cudaFreeHost(', host_code)
    host_code = re.sub(
        r'cudaMemcpy\s*\([^;]+cudaMemcpyHostToDevice[^;]+;',
        '// removed for host memory', host_code
    )
    host_code = re.sub(
        r'cudaMemcpy\s*\([^;]+cudaMemcpyDeviceToHost[^;]+;',
        '// removed for host memory', host_code
    )

    # UNIFIED version
    unified_code = re.sub(
        r'cudaMalloc\s*\(\s*\(void\s*\*\*\)\s*&',
        'cudaMallocManaged((void **)&',
        original
    )
    unified_code = re.sub(
        r'cudaMemcpy\s*\([^;]+cudaMemcpyHostToDevice[^;]+;',
        '// removed for unified memory', unified_code
    )
    unified_code = re.sub(
        r'cudaMemcpy\s*\([^;]+cudaMemcpyDeviceToHost[^;]+;',
        '// removed for unified memory', unified_code
    )

    return {
        'device':  original,
        'host':    host_code,
        'unified': unified_code
    }

def compile_version(kernel_dir, source_file, mem_type, modified_code, binary_name):
    backup = source_file + ".backup"
    shutil.copy(source_file, backup)  # always snapshot current (instrumented) source

    with open(source_file, 'w') as f:
        f.write(modified_code)

    output_bin = f"{binary_name}_{mem_type}"
    result = subprocess.run(
        f"make 2>&1 && mv {binary_name} {output_bin}",
        shell=True, cwd=kernel_dir,
        capture_output=True, text=True, timeout=60
    )

    shutil.copy(backup, source_file)

    if result.returncode == 0:
        return output_bin, None
    else:
        return None, result.stdout + result.stderr

def run_and_time(kernel_dir, binary, args):
    cmd = f"./{binary} {args} 2>&1"
    try:
        result = subprocess.run(
            cmd, shell=True, cwd=kernel_dir,
            capture_output=True, text=True, timeout=60
        )
        output = result.stdout + result.stderr
        patterns = [
            (r'(?:Total k|K)ernel(?: execution)? time\D+(\d+\.?\d*)\s*\(us\)', 1.0),
            (r'(?:Total k|K)ernel(?: execution)? time[^\n=]*[=: ]+(\d+\.?\d*)\s*\(s\)', 1e6),
            (r'Kernel time\s*=\s*(\d+\.?\d+)\s*\(us\)', 1.0),
            (r'offloading time\s*=\s*(\d+\.?\d+)\s*\(s\)', 1e6),
            (r'Average kernel execution time:\s*(\d+\.?\d+)\s*\(s\)', 1e6),
            (r'Average execution time[^:]*:\s*(\d+\.?\d+)\s*\(us\)', 1.0),
            (r'[Kk]ernel time[^:]*:\s*(\d+\.?\d+)\s*ms', 1000.0),
            (r'[Tt]otal time[^:]*:\s*(\d+\.?\d+)\s*ms', 1000.0),
            (r'[Ee]lapsed[^:]*:\s*(\d+\.?\d+)\s*ms', 1000.0),
            (r'(\d+\.?\d+)\s*\(us\)', 1.0),
            (r'(\d+\.?\d+)\s*\(s\)', 1e6),
        ]

        for pattern, multiplier in patterns:
            matches = re.findall(pattern, output)
            if matches:
                val = float(matches[0]) * multiplier
                return val, output

        return None, output
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"

def ask_ollama(kernel_name, device_time, host_time, unified_time):
    prompt = f"""You are a GPU performance expert analyzing CUDA kernel benchmarks.

Kernel: '{kernel_name}'
Timing results:
- Device memory : {device_time} microseconds
- Host memory   : {host_time} microseconds
- Unified memory: {unified_time} microseconds

Answer briefly:
1. Which memory type is fastest?
2. Why did it perform best?
3. Recommendation for this kernel?
"""
    try:
        import requests, json
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3", "prompt": prompt, "stream": False},
            timeout=120
        )
        return response.json()["response"]
    except Exception as e:
        return f"Ollama error: {e}"

def main():
    print(f"Starting auto-modifier benchmark of {len(KERNELS)} kernels...\n")

    all_results = []

    with open(RECOMMENDATIONS, 'w') as f:
        f.write("Batch Benchmark Recommendations\n")
        f.write("=" * 50 + "\n")

    for idx, (kernel, (binary, args)) in enumerate(KERNELS.items(), 1):
        print(f"\n[{idx}/{len(KERNELS)}] {kernel}")
        print("=" * 50)

        kernel_dir = os.path.join(SRC_BASE, kernel)
        if not os.path.exists(kernel_dir):
            print(f"  SKIPPING - directory not found")
            continue

        source_file = find_main_cu(kernel_dir)
        if not source_file:
            print(f"  SKIPPING - no .cu source found")
            continue

        print(f"  Source: {os.path.basename(source_file)}")

        versions = create_memory_versions(source_file)
        timings = {}

        for mem_type, modified_code in versions.items():
            print(f"  [{mem_type}] Compiling...", end=" ", flush=True)
            bin_path, err = compile_version(
                kernel_dir, source_file, mem_type, modified_code, binary
            )
            if not bin_path:
                print(f"FAILED")
                timings[mem_type] = 9999.0
                continue

            print(f"Running...", end=" ", flush=True)
            time_val, output = run_and_time(kernel_dir, bin_path, args)

            if time_val:
                timings[mem_type] = round(time_val, 3)
                print(f"{time_val:.3f} us")
            else:
                timings[mem_type] = 9999.0
                print(f"timing not found")

        winner = min(timings, key=timings.get)

        print(f"\n  Results:")
        print(f"    Device  : {timings.get('device')} us")
        print(f"    Host    : {timings.get('host')} us")
        print(f"    Unified : {timings.get('unified')} us")
        print(f"    Winner  : {winner.upper()}")

        print(f"  Asking llama3...", flush=True)
        rec = ask_ollama(kernel,
                         timings.get('device'),
                         timings.get('host'),
                         timings.get('unified'))

        with open(RECOMMENDATIONS, 'a') as f:
            f.write(f"\nKernel: {kernel}\n")
            f.write(f"Device: {timings.get('device')}us | Host: {timings.get('host')}us | Unified: {timings.get('unified')}us\n")
            f.write(f"Winner: {winner.upper()}\n")
            f.write(f"Llama3: {rec}\n")
            f.write("-" * 50 + "\n")

        all_results.append({
            'kernel': kernel,
            'device_us': timings.get('device'),
            'host_us': timings.get('host'),
            'unified_us': timings.get('unified'),
            'winner': winner
        })
        print(f"  Saved!")

    print(f"\n{'=' * 50}")
    print(f"BATCH COMPLETE!")
    print(f"{'=' * 50}\n")

    with open(RESULTS_CSV, 'w') as f:
        f.write("kernel,device_us,host_us,unified_us,winner\n")
        for r in all_results:
            f.write(f"{r['kernel']},{r['device_us']},{r['host_us']},{r['unified_us']},{r['winner']}\n")

    print("Final Results:")
    print("kernel,device_us,host_us,unified_us,winner")
    for r in all_results:
        print(f"{r['kernel']},{r['device_us']},{r['host_us']},{r['unified_us']},{r['winner']}")

if __name__ == "__main__":
    main()