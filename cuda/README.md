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
Paralelisasi ini mengubah perhitungan nested loop menjadi kernel CUDA. Sebagai eksperimen, terdapat 

Example:

```cpp
// Serial version loop
for (int y = 1; y < in.h - 1; y++) {
    for (int x = 1; x < in.w - 1; x++) {
        ...
    }
}
```
With this
``` cpp
// Parallel version with MPI (row distribution example)
int rowsPerProcess = in.h / world_size;
int startRow = rank * rowsPerProcess;
int endRow   = (rank == world_size - 1) ? in.h - 1 : (startRow + rowsPerProcess);

for (int y = startRow; y < endRow; y++) {
    for (int x = 1; x < in.w - 1; x++) {
        ...
    }
}
```
You may also include screenshots of modified sections and highlight the differences.


## 4. Results and Evaluation

### 4.1 Correctness
- Did the parallel version produce the same output image as the serial version?  
- Show example input and output images for both versions.  

Example (replace with actual images):

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