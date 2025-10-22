# Parallelization Report — Sobel Edge Detection with CUDA

## Team Information
- **Team ID :**  barracuda
- **Class   :**  K3

### Members
| Name                      | Student ID      |
|---------------------------|-----------------|
| Ahmad Wafi Idzharulhaq    | 13523131        |
| Muhamad Nazih Najmudin    | 13523144        |
| Lukas Raja Agripa         | 13523158        |



## List of Contents
1. [Introduction](#1-introduction)  
2. [Theory: Parallelizable Operations](#2-theory-parallelizable-operations)  
3. [Code Changes and Implementation](#3-code-changes-and-implementation)  
4. [Results and Evaluation](#4-results-and-evaluation)  
   - [Correctness](#41-correctness)  
   - [Performance Comparison](#42-performance-comparison)  
   - [Speedup and Efficiency](#43-speedup-and-efficiency)  
5. [Discussion](#5-discussion)  
6. [Conclusion](#6-conclusion)  
7. [Additional Notes (Optional)](#7-additional-notes-optional)  
8. [References](#8-references)  
9. [How to Run](#9-how-to-run)  



## 1. Introduction
Tugas ini bertujuan untuk mengimplementasikan paralelisasi algoritma Sobel Edge Detection menggunakan CUDA. Sobel Edge Detection adalah algoritma pengolahan citra yang digunakan untuk mendeteksi tepi (edge) pada gambar dengan menerapkan operator konvolusi menggunakan kernel Sobel.

Algoritma Sobel bekerja dengan menghitung gradien intensitas gambar menggunakan dua kernel 3x3 untuk mendeteksi perubahan horizontal dan vertikal. Hasil dari kedua gradien tersebut dikombinasikan untuk menghasilkan magnitude gradien yang kemudian dibandingkan dengan threshold untuk menentukan apakah suatu pixel merupakan tepi atau bukan.

Tujuan dari penerapan paralelisasi dengan CUDA adalah untuk meningkatkan performa komputasi dengan memanfaatkan GPU. CUDA memungkinkan pembagian beban kerja komputasi setiap pixel secara paralel, sehingga harapannya waktu pemrosesan dapat berkurang secara signifikan dibandingkan dengan implementasi serial.


## 2. Theory: Parallelizable Operations
Pada kasus ini, operasi dan fungsi yang dapat diparalelisasi adalah:
- Perhituangn konvolusi piksel-piksel dalam algoritma Sobel bersifat independen dan dapat didistribusikan ke seluruh proses.  
- Perhitunagn gradien dan threshold dapat dihitung secara paralel karena setiap piksel bersifat independen.  
- Grayscaling gambar sebetulnya juga dapat diparalelisasi, tetapi kata asisten harus pakai cv::IMREAD_GRAYSCALE


## 3. Code Changes and Implementation
### 3.1 Parallelization Strategy
Paralelisasi dengan CUDA bekerja dengan membagi setiap pemrosesan sebuah piksel gambar ke dalam thread-thread berbeda. Setiap pemrosesan piksel bersifat independen, sehingga seluruh proses mencakup konvolusi dan perhitungan gradien untuk setiap piksel dilakukan pada setiap CUDA thread. 

Strategi Optimasi Overhead:
- Mencari ukuran dimensi block dan dimensi grid yang tepat berdasarkan kemampuan GPU user.
- Pembagian block mengikuti aturan persegi dengan ukuran kelipatan warp size.
- Penyalinan data dari CPU ke GPU relatif mahal, sehingga cudaMalloc dipanggil se-sedikit mungkin. Selain itu juga dioptimasi dengan pinned memory.
- Akses global memory diminimasi dengan membuat shared memory.

### 3.2 Code Modifications
Paralelisasi ini mengubah perhitungan nested loop menjadi kernel CUDA. Serta mengubah perhitungan konvolusi menjadi unroll version.

Before:
```cpp
// Serial version loop
    int Gx[3][3]={{-1,0,1},{-2,0,2},{-1,0,1}};
    int Gy[3][3]={{1,2,1},{0,0,0},{-1,-2,-1}};
    Image out=in;

    for(int y=1;y<in.h-1;y++){
        for(int x=1;x<in.w-1;x++){
            int sx=0, sy=0;
            for(int ky=-1; ky<=1; ky++)
                for(int kx=-1; kx<=1; kx++){
                    int px=in.at(x+kx,y+ky);
                    sx += px * Gx[ky+1][kx+1];
                    sy += px * Gy[ky+1][kx+1];
                }
            int g = std::sqrt(sx*sx + sy*sy);

            ...
        }
    }
```
After:
``` cpp
// Parallel version with CUDA (sobel_kernel without shared memory)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    x++; y++;

    // Convolution
    int ver = 0, hor = 0;
    if (x > 0 && x < w-1 && y > 0 && y < h-1) {
        ver = in[(y+1)*w + (x-1)] + 2*in[(y+1)*w + x] + in[(y+1)*w + (x+1)]
        - (in[(y-1)*w + (x-1)] + 2*in[(y-1)*w + x] + in[(y-1)*w + (x+1)]);
        hor = in[(y-1)*w + (x+1)] + 2*in[y*w + (x+1)] + in[(y+1)*w + (x+1)]
        - (in[(y-1)*w + (x-1)] + 2*in[y*w + (x-1)] + in[(y+1)*w + (x-1)]);
    }
    int g = (int)sqrtf(ver*ver + hor*hor);
```

Contoh diatas merupakan perubahan untuk tipe kernel cuda tanpa shared memory. Untuk implementasi shared memory, terdapat dua versi yaitu versi dengan static block size (16 x 16) dan versi dengan runtime block size (ditentukan oleh fungsi get_optimal_config yang mencari konfigurasi dimensi grid dan blok terbaik sesuai spesifikasi GPU)

Static Block Sized Shared Memory Kernel:
``` cpp
__global__ void kernel_sobel_shared(unsigned char *in, unsigned char *out, int w, int h, int mode, int *d_thresholds) {
    __shared__ unsigned char tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * BLOCK_SIZE + tx;
    int y = blockIdx.y * BLOCK_SIZE + ty;

    // Global index
    int gx = min(max(x, 0), w - 1);
    int gy = min(max(y, 0), h - 1);

    // Load main pixel
    tile[ty + 1][tx + 1] = in[gy * w + gx];

    // Load halo region (edges of tile)
    if (tx == 0 && gx > 0)
        tile[ty + 1][0] = in[gy * w + gx - 1];
    if (tx == BLOCK_SIZE - 1 && gx < w - 1)
        tile[ty + 1][BLOCK_SIZE + 1] = in[gy * w + gx + 1];
    if (ty == 0 && gy > 0)
        tile[0][tx + 1] = in[(gy - 1) * w + gx];
    if (ty == BLOCK_SIZE - 1 && gy < h - 1)
        tile[BLOCK_SIZE + 1][tx + 1] = in[(gy + 1) * w + gx];

    // Load corners (optional, for correctness)
    if (tx == 0 && ty == 0 && gx > 0 && gy > 0)
        tile[0][0] = in[(gy - 1) * w + gx - 1];
    if (tx == BLOCK_SIZE - 1 && ty == 0 && gx < w - 1 && gy > 0)
        tile[0][BLOCK_SIZE + 1] = in[(gy - 1) * w + gx + 1];
    if (tx == 0 && ty == BLOCK_SIZE - 1 && gx > 0 && gy < h - 1)
        tile[BLOCK_SIZE + 1][0] = in[(gy + 1) * w + gx - 1];
    if (tx == BLOCK_SIZE - 1 && ty == BLOCK_SIZE - 1 &&
        gx < w - 1 && gy < h - 1)
        tile[BLOCK_SIZE + 1][BLOCK_SIZE + 1] = in[(gy + 1) * w + gx + 1];

    __syncthreads();

    // Skip border threads
    if (x >= w || y >= h) return;

    // Sobel filter (pakai tile)
    int ver =
        tile[ty + 2][tx] + 2 * tile[ty + 2][tx + 1] + tile[ty + 2][tx + 2] -
        (tile[ty][tx] + 2 * tile[ty][tx + 1] + tile[ty][tx + 2]);
    int hor =
        tile[ty][tx + 2] + 2 * tile[ty + 1][tx + 2] + tile[ty + 2][tx + 2] -
        (tile[ty][tx] + 2 * tile[ty + 1][tx] + tile[ty + 2][tx]);
    int g = (int)sqrtf(ver * ver + hor * hor);
```
Optimize Block Sized Shared Memory Kernel:
``` cpp
__global__ void kernel_sobel_tiled(const unsigned char* __restrict__ in, unsigned char* out,
                        int w, int h, int mode, const int* d_thresholds) {
    const int BX = blockDim.x;
    const int BY = blockDim.y;
    const int sW = BX + 2;   // shared width (cols)
    const int sH = BY + 2;   // shared height (rows)
    extern __shared__ unsigned char s[]; // size: sW * sH

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int x = bx * BX + tx;  // global x for this thread
    int y = by * BY + ty;  // global y for this thread

    int tid = ty * BX + tx;
    int totalThreads = BX * BY;
    int elems = sW * sH;

    // Linearized load (each thread loads multiple elements of shared tile)
    for (int i = tid; i < elems; i += totalThreads) {
        int sm_y = i / sW;
        int sm_x = i % sW;
        int glob_x = bx * BX + (sm_x - 1); // -1 because sm_x=0 is left
        int glob_y = by * BY + (sm_y - 1); // -1 because sm_y=0 is top

        // clamp to image border
        glob_x = (glob_x < 0) ? 0 : ( (glob_x >= w) ? (w-1) : glob_x );
        glob_y = (glob_y < 0) ? 0 : ( (glob_y >= h) ? (h-1) : glob_y );

        s[i] = in[glob_y * w + glob_x];
    }

    __syncthreads();

    if (x >= w || y >= h) return;

    // center in shared coords
    int c_x = tx + 1;
    int c_y = ty + 1;
    // int idx_center = c_y * sW + c_x;

    int ver =
        s[(c_y+1)*sW + (c_x-1)] + 2*s[(c_y+1)*sW + c_x] + s[(c_y+1)*sW + (c_x+1)]
      - (s[(c_y-1)*sW + (c_x-1)] + 2*s[(c_y-1)*sW + c_x] + s[(c_y-1)*sW + (c_x+1)]);
    int hor =
        s[(c_y-1)*sW + (c_x+1)] + 2*s[c_y*sW + (c_x+1)] + s[(c_y+1)*sW + (c_x+1)]
      - (s[(c_y-1)*sW + (c_x-1)] + 2*s[c_y*sW + (c_x-1)] + s[(c_y+1)*sW + (c_x-1)]);

    int g = (int)sqrtf((float)(ver*ver + hor*hor));
```

## 4. Results and Evaluation

### 4.1 Correctness

| Input Image | Serial Output | Parallel Output |
|-------------|---------------|-----------------|
| ![input](path/to/input.jpg) | ![serial](path/to/serial_output.jpg) | ![parallel](path/to/parallel_output.jpg) |



### 4.2 Performance Comparison

#### Serial Version
| Image Name | Input Time (ms) | Processing Time (ms) | Output Time (ms) | Total Time (ms) |
|------------|-----------------|-----------------------|------------------|-----------------|
| image1.jpg |                 |                       |                  |                 |
| image2.jpg |                 |                       |                  |                 |
| image3.jpg |                 |                       |                  |                 |

#### Parallel Version
| Image Name | Core Number | Input Time (ms) | Processing Time (ms) | Output Time (ms) | Total Time (ms) |
|------------|-------------|-----------------|-----------------------|------------------|-----------------|
| image1.jpg | 2           |                 |                       |                  |                 |
| image1.jpg | 4           |                 |                       |                  |                 |
| image1.jpg | 8           |                 |                       |                  |                 |
| image2.jpg | 2           |                 |                       |                  |                 |
| image2.jpg | 4           |                 |                       |                  |                 |
| image2.jpg | 8           |                 |                       |                  |                 |



### 4.3 Speedup and Efficiency
- **Speedup** = Serial Time / Parallel Time  
- **Efficiency** = Speedup / Number of Processes  



## 5. Discussion
- What worked well in your parallelization approach?  
- What challenges did you face (data distribution, communication, synchronization)?  
- Did you notice any overhead, and how did it affect performance?  



## 6. Conclusion
Summarize your findings:  
- Was parallelization effective?  
- Did it improve performance significantly?  
- Any tradeoffs between computation speed and communication overhead?  



## 7. Additional Notes (Optional)
- Suggestions for further improvement (e.g., using hybrid MPI + OpenMP).  
- Observations for different input sizes (small vs large images).  
- Any optimizations beyond the basic parallelization.  



## 8. References
List any references you used (books, lecture notes, research papers, or online sources).



## 9. How to Run
How to run the programs (compiling and running process must atleast)