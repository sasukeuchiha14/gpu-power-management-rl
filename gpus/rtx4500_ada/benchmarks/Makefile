# Use 'nvcc' from the system path
CC := nvcc

# Path to CUDA include and lib (using our symbolic link /usr/local/cuda)
CUDA_DIR := /usr/local/cuda
INCLUDE := $(CUDA_DIR)/include
CUDA_LIB_DIR := $(CUDA_DIR)/lib64

SRC = gaussian.cu
EXE = gaussian

# Optimization flags for modern GPUs (RTX 4500 uses sm_89, but sm_70+ is safe)
FLAGS = -O3

release: $(SRC)
	$(CC) $(FLAGS) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) -lcudart

clean:
	rm -f $(EXE)
