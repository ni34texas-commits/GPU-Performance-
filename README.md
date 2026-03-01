# GPU-Performance
Login Steps:
Terminal
ssh username@zeus.cs.txstate.edu
Enter Password
ssh username@ada.cs.txstate.edu
Enter Password
	Linux commands: 
    ls
    mkdir folderName
    cd folderName
	..
	Create a Python virtual environment:
	python3 -m venv venv
	. venv/bin/activate
	...
	Open a new terminal and install Ollama on Mac/Windows: 
	MAC ->  brew install ollama
	Connect Mac ollama with Zeus and Ada tunnel:
	Mac terminal: ssh -N -J n_i34@zeus.cs.txstate.edu -R 11434:localhost:11434 n_i34@ada
	





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
