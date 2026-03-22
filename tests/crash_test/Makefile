# Makefile for CUDA Crash Test Program
#
# Build: make
# Run: ./cuda_crash_test <test_number>
#
# Requirements:
#   - NVIDIA CUDA Toolkit (nvcc)
#   - CUDA-capable GPU

# Compiler
NVCC = nvcc

# CUDA architecture (adjust for your GPU)
# Examples: sm_70 (V100), sm_80 (A100), sm_89 (RTX 4090), sm_90 (H100)
CUDA_ARCH ?= sm_80

# Compiler flags
NVCC_FLAGS = -arch=$(CUDA_ARCH) -O2 -lineinfo
NVCC_FLAGS += -g -G  # Include debug symbols

# Libraries
LIBS = -lcudart

# Source files
SRC = cuda_crash_test.cu
TARGET = cuda_crash_test

# Default target
all: $(TARGET)

# Build executable
$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(LIBS)

# Build with compute-sanitizer support
debug: NVCC_FLAGS += -G -lineinfo
debug: $(TARGET)

# Create coredump (for testing cuda-gdb-cli)
coredump: $(TARGET)
	./$(TARGET) 1 || true

# Run compute-sanitizer
sanitize: $(TARGET)
	compute-sanitizer ./$(TARGET) 1

# Run with cuda-memcheck
memcheck: $(TARGET)
	cuda-memcheck ./$(TARGET) 1

# Run specific test
test1: $(TARGET)
	./$(TARGET) 1

test2: $(TARGET)
	./$(TARGET) 2

test3: $(TARGET)
	./$(TARGET) 3

test4: $(TARGET)
	./$(TARGET) 4

test5: $(TARGET)
	./$(TARGET) 5

test6: $(TARGET)
	./$(TARGET) 6

# Run all tests
test-all: $(TARGET)
	./$(TARGET) all

# Clean build artifacts
clean:
	rm -f $(TARGET)
	rm -f core.*
	rm -f *.cubin *.fatbin

# Help
help:
	@echo "CUDA Crash Test Program"
	@echo ""
	@echo "Build targets:"
	@echo "  make          - Build the test program"
	@echo "  make debug    - Build with debug symbols"
	@echo "  make clean    - Remove build artifacts"
	@echo ""
	@echo "Test targets:"
	@echo "  make test1    - Run illegal memory access test"
	@echo "  make test2    - Run shared memory race test"
	@echo "  make test3    - Run warp divergence test"
	@echo "  make test4    - Run null pointer test"
	@echo "  make test5    - Run stack overflow test"
	@echo "  make test6    - Run matrix multiplication test"
	@echo "  make test-all - Run all tests"
	@echo ""
	@echo "Debugging targets:"
	@echo "  make coredump - Create coredump for testing"
	@echo "  make sanitize - Run with compute-sanitizer"
	@echo "  make memcheck - Run with cuda-memcheck"
	@echo ""
	@echo "CUDA Architecture:"
	@echo "  Set CUDA_ARCH variable, e.g.:"
	@echo "  make CUDA_ARCH=sm_89"
	@echo ""
	@echo "Examples:"
	@echo "  sm_70  - Tesla V100, GTX 1080"
	@echo "  sm_75  - Tesla T4, RTX 2080"
	@echo "  sm_80  - A100"
	@echo "  sm_86  - RTX 3090, A6000"
	@echo "  sm_89  - RTX 4090, L40"
	@echo "  sm_90  - H100"

.PHONY: all debug clean test1 test2 test3 test4 test5 test6 test-all help coredump sanitize memcheck