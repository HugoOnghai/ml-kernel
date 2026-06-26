# ml-kernel

A CNN inference kernel library built from scratch in C, developed across four progressive optimization labs for Cornell Tech's Modern Computer Systems & Architecture course. Each lab added a new layer of hardware-aware optimization on top of the previous.

## What's Implemented

| Module | Operations |
|--------|-----------|
| `kernel/functional.c` | ReLU, numerically stable softmax |
| `kernel/linear.c` | Fully-connected (dense) layer |
| `kernel/matrix_ops.c` | Naive matmul, cache-blocked matmul, INT8 quantized matmul, CSR sparse matmul, multithreaded matmul |
| `kernel/conv.c` | Direct 2D convolution, im2col-accelerated convolution |
| `utils/data_utils.c` | HDF5 model weight loader, MNIST image/label reader |

## Lab Progression

### Lab 1 — Baseline Primitives
Implemented ReLU, numerically stable log-softmax, a fully-connected linear layer, and direct 2D convolution in C. Profiled with `perf` at multiple input scales to establish baseline performance.

### Lab 2 — Cache Optimization, Quantization, and Attention
- **Cache-blocked matmul**: Tiled the triple loop with a block size of 224, empirically tuned to fit the L1/L2 cache hierarchy of the target Intel Xeon Silver. Dramatically reduces cache misses vs. the naive traversal order.
- **INT8 quantized matmul**: Packs 4 INT8 values per `uint32_t` (big-endian), reducing memory bandwidth by 4×. Uses bit-shifting to unpack values at multiply time.
- **Scaled dot-product attention**: Full Transformer attention (`Q·Kᵀ / √dₖ → softmax → × V`) implemented in C.

### Lab 3 — Sparse MatMul and Im2col Convolution
- **CSR sparse matmul**: Converts dense matrices to Compressed Sparse Row format and performs SpGEMM, skipping all zero-element multiplications. Benchmarked at sparsity levels 0.1, 0.5, and 0.9.
- **Im2col convolution**: Reformulates 2D convolution as a single GEMM call — flattens the kernel weights and maps input patches into columns, then dispatches to any matmul backend (dense or sparse).

### Lab 4 — Multithreaded MatMul
- Spawns 8 POSIX threads matched to the 8-core Slurm allocation on the Intel Xeon Silver 4514Y cluster.
- Allocates the output matrix with `posix_memalign(..., 64, ...)` to align to 64-byte cache lines, eliminating false sharing between threads.
- Divides work by cache lines and distributes evenly across threads.
- Profiled with Intel Top-Down Methodology (`toplev.py`) — the naive version shows 39.8% `Backend_Bound.Core_Bound`; the threaded version achieves 58.4% `Retiring`, confirming effective core utilization.

**Speedup results (1300×1300 matrix multiply, Intel Xeon Silver 4514Y):**

| Matrix Size | Naive | Threaded (8 cores) | Speedup |
|-------------|-------|--------------------|---------|
| 10×10       | 0.256s | 0.212s            | 1.2×    |
| 100×100     | 0.251s | 0.220s            | 1.1×    |
| 200×200     | 0.388s | 0.271s            | 1.4×    |
| 500×500     | 2.664s | 0.509s            | 5.2×    |
| 1000×1000   | 24.426s | 2.397s           | 10.2×   |
| 2000×2000   | 227.948s | 17.890s         | 12.7×   |

## Build

Requires `gcc`, `libhdf5`, and `libm`.

```bash
# Run the test suite
make test

# Build and profile multithreaded vs naive matmul (requires Intel pmu-tools)
make matmul_naive
make matmul_thread

# Clean build artifacts
make clean
```

## Tech

C · POSIX Threads · HDF5 · Intel Top-Down Methodology (toplev) · Unity test framework
