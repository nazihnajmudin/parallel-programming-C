# Parallelization of Sobel Edge Detection

## Project Overview

This project implements and compares four different parallelization approaches for the Sobel Edge Detection algorithm. Sobel Edge Detection is a fundamental image processing technique used to detect edges by computing gradient approximations of image intensity. The parallelization strategies explored include:

1. **AVX2 SIMD** - Single Instruction, Multiple Data vectorization
2. **CUDA GPU** - Graphics Processing Unit acceleration
3. **OpenMP** - Shared-memory multi-threading
4. **OpenMPI** - Distributed-memory parallel processing

Each implementation maintains functional correctness while varying in performance characteristics, scalability, and hardware requirements.

## Team Information

- **Team ID:** barracuda  
- **Class:** K3

### Members

| Name | Student ID |
|-----------------------------|------------|
| Ahmad Wafi Idzharulhaq | 13523131 |
| Muhamad Nazih Najmudin | 13523144 |
| Lukas Raja Agripa | 13523158 |

## Parallelization Approaches

| Approach | Technology | Hardware | Granularity | Best Use Case |
|----------|------------|----------|-------------|---------------|
| **AVX2** | SIMD intrinsics | CPU | 8 pixels | Small to medium images, CPU-bound systems |
| **CUDA** | GPU computing | NVIDIA GPU | Thread per pixel | Large images, compute-intensive workloads |
| **OpenMP** | Shared memory | CPU (multi-core) | Row-wise | Medium-large images, shared memory systems |
| **OpenMPI** | Distributed memory | CPU cluster | Row-wise | Large images, distributed systems |

## Results Summary

### AVX2

| Image | Serial Total (ms) | Parallel Total (ms) | Speedup |
|-------|-------------------|---------------------|---------|
| birds.jpg | 365 | 78 | 4.68× |
| fish.jpg | 402 | 387 | 1.04× |
| lion.jpg | 62 | 62 | 1.00× |
| snake.jpg | 128 | 111 | 1.15× |
| view.jpg | 148 | 136 | 1.09× |

AVX2 provides the highest speedup (4.68×) for small images with minimal overhead. Performance gains are limited by I/O and memory bandwidth for larger images.

### CUDA

| Image | Kernel Type | Serial Proc (ms) | Parallel Proc (ms) | Speedup (Proc) |
|-------|-------------|------------------|--------------------|----------------|
| fish.jpg | raw-s | 477 | 16 | 29.81× |
| fish.jpg | shared-d | 477 | 14 | 34.07× |
| view.jpg | shared-d | 128 | 6 | 21.33× |

CUDA achieves the highest processing speedup (up to 34×), but total speedup is limited by data transfer and I/O overhead.

### OpenMP

| Image | Threads | Serial Total (ms) | Parallel Total (ms) | Speedup |
|-------|---------|-------------------|---------------------|---------|
| fish.jpg | 8 | 3160 | 754 | 4.19× |
| lion.jpg | 8 | 400 | 136 | 2.94× |
| view.jpg | 8 | 964 | 408 | 2.36× |

OpenMP shows excellent scalability for computation-heavy workloads with 8 threads, though small images suffer from thread overhead.

### OpenMPI

| Image | Processes | Serial Total (ms) | Parallel Total (ms) | Speedup |
|-------|-----------|-------------------|---------------------|---------|
| fish.jpg | 4 | 3160 | 1371 | 2.31× |
| lion.jpg | 8 | 400 | 140 | 2.86× |
| view.jpg | 2 | 964 | 422 | 2.28× |

OpenMPI provides consistent speedup with good scalability up to 4 processes, with diminishing returns at 8 processes due to communication overhead.

## Directory Structure

```
./
├── README.md                   # This file - main documentation
├── test_cases/                 # Input images
│   ├── birds.jpg
│   ├── fish.jpg
│   ├── lion.jpg
│   ├── snake.jpg
│   └── view.jpg
├── serial/                     # Serial reference implementation
│   ├── serial.cpp
│   ├── doc/                    # Performance logs
│   └── pic/                    # Output images
├── avx2/                       # AVX2 SIMD implementation
│   ├── README.md               # Detailed AVX2 report
│   ├── avx2_sobel.cpp
│   ├── avx2_sobel.hpp
│   ├── main.cpp
│   └── output/
│       ├── jpg/                # Output images
│       └── txt/                # Performance logs
├── cuda/                       # CUDA GPU implementation
│   ├── README.md               # Detailed CUDA report
│   ├── header/
│   ├── main.cu
│   ├── makefile
│   └── output/
│       ├── jpg/                # Output images
│       └── txt/                # Performance logs
├── open_mp/                    # OpenMP implementation
│   ├── README.md               # Detailed OpenMP report
│   ├── header/
│   ├── main.cpp
│   ├── mp
│   └── output/
│       ├── jpg/                # Output images
│       └── txt/                # Performance logs
└── open_mpi/                   # OpenMPI implementation
    ├── README.md               # Detailed OpenMPI report
    ├── openmpi.cpp
    ├── sobel_mpi
    ├── doc/                    # Performance logs
    └── pic/                    # Output images
```

## How to Run

### Prerequisites

- **AVX2**: CPU with AVX2 support, OpenCV 4.x, GCC/G++ with C++17
- **CUDA**: NVIDIA GPU, CUDA Toolkit 12.8+, OpenCV 4.x
- **OpenMP**: Compiler with OpenMP support (`-fopenmp`)
- **OpenMPI**: OpenMPI library, OpenCV 4.x

### Compilation

#### AVX2
```bash
cd avx2
g++ -g -mavx2 main.cpp avx2_sobel.cpp -o avx2_sobel `pkg-config --cflags --libs opencv4`
```

#### CUDA
```bash
cd cuda
make
chmod +x barracuda
```

#### OpenMP
```bash
cd open_mp
g++ -fopenmp -O2 -o mp main.cpp mp.cpp -lm
```

#### OpenMPI
```bash
cd open_mpi
mpic++ -o sobel_mpi openmpi.cpp `pkg-config --cflags --libs opencv4`
```

### Execution

#### AVX2
```bash
cd avx2
./avx2_sobel <mode> <input.jpg> <output.jpg> > test.txt
```

#### CUDA
```bash
cd cuda
./barracuda [raw|shared] [d|s] <mode> <input.jpg> <output.jpg> > output.txt
```

#### OpenMP
```bash
cd open_mp
./mp <num_threads> <input_image> <output_image> > log.txt
```

#### OpenMPI
```bash
cd open_mpi
mpirun -np <num_processes> --oversubscribe ./sobel_mpi <mode> <input.jpg> <output.jpg> > log.txt
```

### Mode Parameter

- `0` — Grayscale Gradient magnitude
- `1` — Binary Threshold output
- `n ≥ 2` — Multi-Level Threshold output

## Detailed Reports

For comprehensive analysis including correctness verification, performance measurements, speedup calculations, and in-depth discussion of each parallelization strategy, please refer to the individual README files in each subdirectory:

- [AVX2 Detailed Report](avx2/README.md)
- [CUDA Detailed Report](cuda/README.md)
- [OpenMP Detailed Report](open_mp/README.md)
- [OpenMPI Detailed Report](open_mpi/README.md)

## References

1. [Sobel Algorithm Tutorial](https://youtu.be/uihBwtPIBxM)
2. [AVX2 Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
3. [CUDA Programming Guide](https://docs.nvidia.com/cuda/)
4. [OpenMP API](https://www.openmp.org/)
5. [OpenMPI Documentation](https://www.open-mpi.org/doc/)
6. Lecture Materials: IF3130 - Parallel and Distributed Systems, Institut Teknologi Bandung