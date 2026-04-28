# AI agent that recommends the best memory location for GPU programs
Step 1: Set Up Your Environment
	Install Cuda Toolkit: developer.nvidia.com
	Install Python3
	Clone: git clone https://github.com/zjin-lcf/HeCBench 
.................
I used the Texas State University's Ada server:
	Login Steps:
	Terminal
		ssh username@zeus.cs.txstate.edu
		Enter Password
		ssh username@ada.cs.txstate.edu
		Enter Password
.......
Linux command: 
	#Check Nvidia Version:
	nvcc --version
	#Check Python3 Version installed (if not, install Python3)
	python3 --version
	Git Clone: git clone https://github.com/zjin-lcf/HeCBench
	
#Step 2: Select a Subset (Histogram-cuda) of HeCBench Kernels
	Linux command: cd histogram-cuda/
				   ls
		           make run (it will provide the GPU's best runtime).
				   ...................................
				   ls
				   #To check the code of histogram-cuda/
				   cat histogram_compare.cu // because we need to include code for CPU and Unified Memory
				   nano histogram_compare.cu //Change the code: 3 memory configuration, such as (Host, Device, and UM) using cudaMalloc, cudaMallocHost, cudaMallocManaged
				   Record the best runtime. 
				   Repeat it: pick 20-50 diverse kernels. 
				   .......................................
#Step 3: Install Ollama and pull codellama:
How do you add the benchmarks to my ground truth database?
The Process to Add a New Kernel
It is a 4-step process:
	Step 1 — Find and instrument the kernel: Example: find cluster.cu, added cudaEvent_t timing
	Step 2 — Benchmark all 3 memory types manually: Example: ./kmeans -i /tmp/kmeans_input.txt -m 5 -n 5
	Step 3 — Record the real timings: Example: Device=116us, Host=203us, Unified=103us → UNIFIED wins
	Step 4 — Add to ground truth dictionary: Example: Add "kmeans-cuda": "unified" to GROUND_TRUTH


	







				   
	
	....................................................
	Create a Python virtual environment:
	python3 -m venv venv
	. venv/bin/activate
	...
	Open a new terminal and install Ollama on Mac/Windows: 
	MAC ->  brew install ollama
	





	..........
    git clone https://github.com/Scientific-Computing-Lab/ParallelCodeEstimation.git
..............
Before running the program make sure these are installed: 
	CMake ✅
	clang / clang++ ✅
	CUDA toolkit (nvcc, cuobjdump) ✅
	Nsight Compute (ncu) ✅

................
Main Dataset Link: https://github.com/Scientific-Computing-Lab/ParallelCodeEstimation 
