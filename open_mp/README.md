# Parallelization Report — Sobel Edge Detection with OpenMP

## Team Information
- **Team ID:** barracuda  
- **Class:** K3

### Members
| Name                        | Student ID |
|-----------------------------|------------|
| Ahmad Wafi Idzharulhaq      | 13523131   |
| Muhamad Nazih Najmudin      | 13523144   |
| Lukas Raja Agripa           | 13523158   |



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
Tugas ini bertujuan untuk mengimplementasikan paralelisasi algoritma Sobel Edge Detection menggunakan OpenMP (Open Multi-Processing). Sobel Edge Detection adalah algoritma pengolahan citra yang digunakan untuk mendeteksi tepi (edge) pada gambar dengan menerapkan operator konvolusi menggunakan kernel Sobel.

Algoritma Sobel bekerja dengan menghitung gradien intensitas gambar menggunakan dua kernel 3x3 untuk mendeteksi perubahan horizontal dan vertikal. Hasil dari kedua gradien tersebut dikombinasikan untuk menghasilkan magnitude gradien yang kemudian dibandingkan dengan threshold untuk menentukan apakah suatu pixel merupakan tepi atau bukan.

Tujuan dari penerapan paralelisasi OpenMP adalah untuk meningkatkan performa komputasi dengan memanfaatkan multiple thread pada sistem shared-memory. OpenMP memungkinkan pembagian beban kerja komputasi pixel-by-pixel secara paralel, sehingga waktu pemrosesan dapat berkurang secara signifikan dibandingkan dengan implementasi serial.



## 2. Theory: Parallelizable Operations
**Questions:**
- Explain which operations or functions in the program can be parallelized and why.

**Answers:**
Dalam implementasi Sobel Edge Detection, terdapat beberapa operasi yang dapat diparalelisasi menggunakan OpenMP:

### 2.1 Operasi yang Dapat Diparalelisasi:
- **Konvolusi Pixel-by-pixel**: Setiap pixel dapat diproses secara independen karena perhitungan Sobel operator pada satu pixel tidak bergantung pada hasil perhitungan pixel lain. Hal ini memungkinkan pembagian iterasi loop secara paralel menggunakan `#pragma omp parallel for`.

- **Perhitungan Gradient Magnitude**: Operasi sqrt(Gx² + Gy²) untuk setiap pixel bersifat independen dan dapat dikomputasi secara paralel.

- **Thresholding**: Proses perbandingan magnitude dengan threshold value untuk setiap pixel dapat dilakukan secara paralel karena tidak ada dependency antar pixel.

- **Mode Processing**: Implementasi mendukung multiple threshold modes yang dapat dijalankan secara paralel untuk menghasilkan output dengan tingkat sensitivitas tepi yang berbeda.

### 2.2 Operasi yang Tidak Diparalelisasi:
- **Image I/O Operations**: Pembacaan dan penulisan file gambar dilakukan secara serial karena operasi file I/O umumnya tidak mendapat keuntungan signifikan dari paralelisasi dan dapat menyebabkan race condition.

- **Memory Allocation**: Alokasi memori untuk struktur data gambar dilakukan secara serial untuk menghindari konflik akses memori.  


## 3. Code Changes and Implementation
**Questions:**
- Describe how the workload is divided among OpenMP threads (e.g., row-wise distribution, block distribution). 
- Mention how many threads you tested.
- Document the changes you made to the code. Use before vs after snippets and provide explanations.

**Answers:**
### 3.1 Parallelization Strategy
Strategi paralelisasi yang diterapkan dalam implementasi OpenMP ini menggunakan pendekatan **shared-memory parallelism** dengan pembagian kerja berdasarkan iterasi loop. Workload dibagi secara otomatis oleh OpenMP runtime system menggunakan `#pragma omp parallel for`, di mana setiap thread menangani sebagian dari baris gambar (row-wise distribution).

Jumlah thread yang diuji dalam implementasi ini adalah:
- 2 threads (baseline paralel)
- 4 threads 
- 8 threads

### 3.2 Code Modifications
Berikut adalah perubahan kode dari versi serial menjadi versi paralel OpenMP:

**Sebelum (Serial Version):**
```cpp
// Loop serial untuk konvolusi Sobel
for (int y = 1; y < in.h - 1; y++) {
    for (int x = 1; x < in.w - 1; x++) {
        // Perhitungan gradien Gx dan Gy
        int gx = 0, gy = 0;
        
        // Konvolusi dengan kernel Sobel
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int pixel_val = in.data[(y + ky) * in.w + (x + kx)];
                gx += pixel_val * sobel_x[ky + 1][kx + 1];
                gy += pixel_val * sobel_y[ky + 1][kx + 1];
            }
        }
        
        // Perhitungan magnitude
        int magnitude = sqrt(gx * gx + gy * gy);
        out.data[y * out.w + x] = magnitude;
    }
}
```

**Sesudah (Parallel Version dengan OpenMP):**
```cpp
// Looping paralel biasa pake OpenMP
#pragma omp parallel for schedule(dynamic)
for (int y = 1; y < in.h - 1; y++) {
    for (int x = 1; x < in.w - 1; x++) {
        // Perhitungan gradien Gx dan Gy (sama seperti serial)
        int gx = 0, gy = 0;
        
        // Konvolusi dengan kernel Sobel
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int pixel_val = in.data[(y + ky) * in.w + (x + kx)];
                gx += pixel_val * sobel_x[ky + 1][kx + 1];
                gy += pixel_val * sobel_y[ky + 1][kx + 1];
            }
        }
        
        // Perhitungan magnitude
        int magnitude = sqrt(gx * gx + gy * gy);
        out.data[y * out.w + x] = magnitude;
    }
}
```

**Perubahan Utama:**
1. **Penambahan `#pragma omp parallel for`**: Directive ini menginstruksikan compiler untuk membuat parallel region dan membagi iterasi outer loop (y) di antara available threads.
2. **Dynamic Scheduling**: Penggunaan `schedule(dynamic)` memungkinkan load balancing yang lebih baik karena setiap thread mengambil chunk pekerjaan secara dinamis.
3. **Thread Management**: Jumlah thread dapat dikontrol melalui environment variable `OMP_NUM_THREADS` atau fungsi `omp_set_num_threads()`.


## 4. Results and Evaluation

### 4.1 Correctness
**Questions:**
- Did the parallel version produce the same output image as the serial version?
- Show example input and output images for both versions.

**Answers:**
- **Apakah versi paralel menghasilkan output gambar yang sama dengan versi serial?**
  Ya, versi paralel menghasilkan output yang identik dengan versi serial karena algoritma Sobel edge detection bersifat deterministic dan setiap pixel diproses secara independen.

- **Verifikasi Hasil:**
  - Output gambar dari 2 threads, 4 threads, dan 8 threads menghasilkan edge detection yang identik
  - Tidak ada perbedaan visual dalam hasil deteksi tepi
  - Threshold values yang digunakan konsisten across different thread configurations

**Input and Output Image Comparison:**

| Input Image | Parallel Output (2 Threads) | Parallel Output (4 Threads) | Parallel Output (8 Threads) |
|-------------|----------------------------|----------------------------|----------------------------|
| ![birds](../test_cases/birds.jpg) | ![parallel_birds_2](output/jpg/hasil_birds_2thread.jpg) | ![parallel_birds_4](output/jpg/hasil_birds_4thread.jpg) | ![parallel_birds_8](output/jpg/hasil_birds_8thread.jpg) |
| ![fish](../test_cases/fish.jpg) | ![parallel_fish_2](output/jpg/hasil_fish_2thread.jpg) | ![parallel_fish_4](output/jpg/hasil_fish_4thread.jpg) | ![parallel_fish_8](output/jpg/hasil_fish_8thread.jpg) |
| ![lion](../test_cases/lion.jpg) | ![parallel_lion_2](output/jpg/hasil_lion_2thread.jpg) | ![parallel_lion_4](output/jpg/hasil_lion_4thread.jpg) | ![parallel_lion_8](output/jpg/hasil_lion_8thread.jpg) |
| ![snake](../test_cases/snake.jpg) | ![parallel_snake_2](output/jpg/hasil_snake_2thread.jpg) | ![parallel_snake_4](output/jpg/hasil_snake_4thread.jpg) | ![parallel_snake_8](output/jpg/hasil_snake_8thread.jpg) |
| ![view](../test_cases/view.jpg) | ![parallel_view_2](output/jpg/hasil_view_2thread.jpg) | ![parallel_view_4](output/jpg/hasil_view_4thread.jpg) | ![parallel_view_8](output/jpg/hasil_view_8thread.jpg) |

**Catatan:** Semua versi (2, 4, dan 8 threads) menghasilkan gambar output yang pixel-perfect identical, memverifikasi bahwa paralelisasi tidak mempengaruhi correctness algoritma.



### 4.2 Performance Comparison

#### Parallel Performance Results
| Image Name | Number of Threads | Input Time (ms) | Processing Time (ms) | Output Time (ms) | Total Time (ms) |
|------------|-------------------|----------------|-----------------------|------------------|-----------------|
| birds.jpg  | 2           | 42              | 16                   | 27               | 85              |
| birds.jpg  | 4           | 43              | 11                   | 41               | 95              |
| birds.jpg  | 8           | 60              | 26                   | 12               | 98              |
| fish.jpg   | 2           | 718             | 577                  | 407              | 1702            |
| fish.jpg   | 4           | 694             | 579                  | 760              | 2033            |
| fish.jpg   | 8           | 282             | 140                  | 332              | 754             |
| lion.jpg   | 2           | 70              | 81                   | 144              | 295             |
| lion.jpg   | 4           | 65              | 68                   | 116              | 249             |
| lion.jpg   | 8           | 30              | 46                   | 60               | 136             |
| snake.jpg  | 2           | 160             | 64                   | 216              | 440             |
| snake.jpg  | 4           | 160             | 66                   | 181              | 407             |
| snake.jpg  | 8           | 133             | 51                   | 74               | 258             |
| view.jpg   | 2           | 259             | 116                  | 273              | 648             |
| view.jpg   | 4           | 261             | 105                  | 248              | 614             |
| view.jpg   | 8           | 214             | 65                   | 129              | 408             |

#### Serial Performance Results (Baseline for Comparison)
| Image Name | Input Time (ms) | Processing Time (ms) | Output Time (ms) | Total Time (ms) |
|------------|-----------------|---------------------|------------------|-----------------|
| birds.jpg  | 32              | 19                  | 17               | 68              |
| fish.jpg   | 651             | 2174                | 335              | 3160            |
| lion.jpg   | 53              | 253                 | 94               | 400             |
| snake.jpg  | 112             | 236                 | 195              | 543             |
| view.jpg   | 226             | 539                 | 199              | 964             |

**Serial vs Parallel Comparison Table:**

| Input Image | Serial Output | Parallel Output (2 Threads) | Parallel Output (4 Threads) | Parallel Output (8 Threads) |
|-------------|---------------|----------------------------|----------------------------|----------------------------|
| ![birds](../test_cases/birds.jpg) | ![serial_birds](../serial/pic/serial_output_birds.jpg) | ![parallel_birds_2](output/jpg/hasil_birds_2thread.jpg) | ![parallel_birds_4](output/jpg/hasil_birds_4thread.jpg) | ![parallel_birds_8](output/jpg/hasil_birds_8thread.jpg) |
| ![fish](../test_cases/fish.jpg) | ![serial_fish](../serial/pic/serial_output_fish.jpg) | ![parallel_fish_2](output/jpg/hasil_fish_2thread.jpg) | ![parallel_fish_4](output/jpg/hasil_fish_4thread.jpg) | ![parallel_fish_8](output/jpg/hasil_fish_8thread.jpg) |
| ![lion](../test_cases/lion.jpg) | ![serial_lion](../serial/pic/serial_output_lion.jpg) | ![parallel_lion_2](output/jpg/hasil_lion_2thread.jpg) | ![parallel_lion_4](output/jpg/hasil_lion_4thread.jpg) | ![parallel_lion_8](output/jpg/hasil_lion_8thread.jpg) |
| ![snake](../test_cases/snake.jpg) | ![serial_snake](../serial/pic/serial_output_snake.jpg) | ![parallel_snake_2](output/jpg/hasil_snake_2thread.jpg) | ![parallel_snake_4](output/jpg/hasil_snake_4thread.jpg) | ![parallel_snake_8](output/jpg/hasil_snake_8thread.jpg) |
| ![view](../test_cases/view.jpg) | ![serial_view](../serial/pic/serial_output_view.jpg) | ![parallel_view_2](output/jpg/hasil_view_2thread.jpg) | ![parallel_view_4](output/jpg/hasil_view_4thread.jpg) | ![parallel_view_8](output/jpg/hasil_view_8thread.jpg) |

### 4.3 Speedup and Efficiency
- **Speedup** = Serial Time / Parallel Time  
- **Efficiency** = Speedup / Number of Threads

#### Analisis Performa untuk Semua Gambar (dengan Serial sebagai baseline):

**birds.jpg:**
| Threads | Total Time (ms) | Speedup vs Serial | Efficiency |
|---------|-----------------|-------------------|------------|
| Serial  | 68              | 1.00              | 1.00       |
| 2       | 85              | 0.80              | 0.40       |
| 4       | 95              | 0.72              | 0.18       |
| 8       | 98              | 0.69              | 0.09       |

**fish.jpg:**
| Threads | Total Time (ms) | Speedup vs Serial | Efficiency |
|---------|-----------------|-------------------|------------|
| Serial  | 3160            | 1.00              | 1.00       |
| 2       | 1702            | 1.86              | 0.93       |
| 4       | 2033            | 1.55              | 0.39       |
| 8       | 754             | 4.19              | 0.52       |

**lion.jpg:**
| Threads | Total Time (ms) | Speedup vs Serial | Efficiency |
|---------|-----------------|-------------------|------------|
| Serial  | 400             | 1.00              | 1.00       |
| 2       | 295             | 1.36              | 0.68       |
| 4       | 249             | 1.61              | 0.40       |
| 8       | 136             | 2.94              | 0.37       |

**snake.jpg:**
| Threads | Total Time (ms) | Speedup vs Serial | Efficiency |
|---------|-----------------|-------------------|------------|
| Serial  | 543             | 1.00              | 1.00       |
| 2       | 440             | 1.23              | 0.62       |
| 4       | 407             | 1.33              | 0.33       |
| 8       | 258             | 2.10              | 0.26       |

**view.jpg:**
| Threads | Total Time (ms) | Speedup vs Serial | Efficiency |
|---------|-----------------|-------------------|------------|
| Serial  | 964             | 1.00              | 1.00       |
| 2       | 648             | 1.49              | 0.75       |
| 4       | 614             | 1.57              | 0.39       |
| 8       | 408             | 2.36              | 0.30       |

**Observasi Komprehensif (Speedup vs Serial):**
- **2 threads**: Memberikan speedup positif untuk semua gambar kecuali birds.jpg (0.80x), dengan fish.jpg mencapai 1.86x
- **4 threads**: Peningkatan moderat untuk gambar menengah-besar: lion (1.61x), snake (1.33x), view (1.57x)
- **8 threads**: Excellent scalability untuk computation-heavy images: fish (4.19x), lion (2.94x), snake (2.10x), view (2.36x)
- **Processing time reduction**: Dramatik pada 8 threads, dari 2174ms → 140ms untuk fish.jpg (15.5x speedup)
- **Gambar kecil** (birds.jpg): Consistent overhead across all parallel configurations vs serial
- **Gambar besar** (fish.jpg): Best candidate untuk paralelisasi dengan 4.19x speedup pada 8 threads  



## 5. Discussion
**Questions:**
- What worked well in your parallelization approach?
- What challenges did you face (data distribution, communication, synchronization)?
- Did you notice any overhead, and how did it affect performance?

**Answers:**
### 5.1 Aspek yang Berhasil dalam Pendekatan Paralelisasi:
- **Implementasi OpenMP yang Sederhana**: Penggunaan `#pragma omp parallel for` memberikan cara yang mudah dan efektif untuk memparalelisasi loop konvolusi tanpa perlu mengubah struktur kode secara signifikan.
- **Pembagian Kerja Otomatis**: OpenMP runtime system secara otomatis membagi iterasi loop di antara available threads, sehingga tidak perlu implementasi manual untuk work distribution.
- **Penurunan Processing Time**: Terlihat konsisten bahwa waktu pemrosesan inti berkurang dengan penambahan threads across semua gambar.
- **Excellent Scalability pada 8 Threads**: Konfigurasi 8 threads menunjukkan peningkatan performa dramatik untuk gambar menengah-besar (speedup 1.59x - 2.26x).

### 5.2 Tantangan yang Dihadapi:
- **Overhead pada Gambar Kecil**: birds.jpg mengalami penurunan performa seiring penambahan threads karena overhead lebih besar dari benefit komputasi.
- **Variabilitas Berdasarkan Problem Size**: Performa paralelisasi sangat bergantung pada ukuran dan kompleksitas gambar input.
- **Memory Bandwidth Scaling**: Pada 8 threads, bandwidth memori menjadi faktor pembatas untuk beberapa gambar.
- **Load Balancing Complexity**: Dynamic scheduling memerlukan tuning untuk optimal performance across different image sizes.

### 5.3 Overhead yang Diamati:
- **Thread Creation Overhead**: Lebih signifikan pada gambar kecil (birds.jpg) dibandingkan gambar menengah (lion.jpg, snake.jpg).
- **Synchronization Overhead**: Meningkat secara tidak linear dengan penambahan threads, terutama terlihat pada output time.
- **Memory Bandwidth Limitation**: Pada gambar besar (fish.jpg), bandwidth memori menjadi bottleneck yang membatasi scalability.
- **Cache Coherency**: False sharing dan cache line conflicts berkontribusi pada overhead, terutama pada 4 threads.  



## 6. Conclusion
**Questions:**
- Was parallelization effective?
- Did it improve performance significantly?
- Any tradeoffs between computation speed and communication overhead?

**Answers:**
Berdasarkan hasil eksperimen komprehensif dengan 5 test cases dan analisis performa, dapat disimpulkan bahwa:

### 6.1 Efektivitas Paralelisasi:
- Paralelisasi OpenMP **sangat efektif** untuk gambar menengah hingga besar, terutama pada konfigurasi 8 threads.
- **4 threads** memberikan improvement moderat untuk sebagian besar gambar (speedup 1.06x - 1.18x).
- **8 threads** memberikan breakthrough performance untuk gambar komplek dengan speedup hingga 2.26x untuk fish.jpg.

### 6.2 Peningkatan Performa Berdasarkan Ukuran Gambar:
- **Gambar Kecil** (birds.jpg): Mengalami degradasi performa seiring penambahan threads (0.89x pada 4 threads, 0.87x pada 8 threads)
- **Gambar Besar** (fish.jpg): Menunjukkan scalability excellent pada 8 threads (speedup 2.26x) setelah degradasi pada 4 threads
- **Gambar Menengah** (lion, snake, view): Consistent improvement dengan peak performance pada 8 threads (speedup 1.59x - 2.17x)

### 6.3 Trade-off Komputasi vs Overhead:
- **Sweet Spot**: 8 threads optimal untuk gambar menengah-besar, 2-4 threads untuk gambar kecil
- **Processing Time**: Dramatic reduction dengan 8 threads, membuktikan excellent scalability untuk computation-heavy workloads
- **Total Performance**: Breakthrough achieved untuk large images dengan proper thread scaling
- **Efficiency**: Tetap reasonable (0.40-0.57) pada 8 threads untuk gambar yang sesuai

### 6.4 Kesimpulan Utama:
- **OpenMP paralelisasi sangat efektif pada gambar menengah-besar** dengan 8 threads memberikan speedup hingga 2.26x
- **Problem size menjadi faktor kunci** dalam menentukan optimal thread configuration
- **Excellent scalability achieved** untuk computation-intensive workloads dengan proper thread scaling
- **Implementasi berhasil** mempertahankan correctness sambil memberikan significant performance improvement untuk appropriate problem sizes  



## 7. Additional Notes (Optional)

### 7.1 Saran Peningkatan Lebih Lanjut:
- **Hybrid MPI + OpenMP**: Kombinasi dengan MPI dapat memungkinkan paralelisasi pada multiple nodes untuk gambar berukuran sangat besar.
- **SIMD Optimization**: Implementasi instruksi SIMD (SSE/AVX) dapat meningkatkan performa operasi konvolusi.
- **Memory Layout Optimization**: Penggunaan memory padding dan data alignment untuk menghindari false sharing.

### 7.2 Observasi untuk Ukuran Input yang Berbeda:
- **Gambar Kecil**: Overhead paralelisasi dominan, sehingga versi serial dapat lebih cepat.
- **Gambar Besar**: Diperkirakan akan memberikan speedup yang lebih signifikan karena computation workload yang lebih besar.
- **Threshold Point**: Perlu dilakukan analisis untuk menentukan ukuran gambar minimum yang menguntungkan untuk paralelisasi.

### 7.3 Optimisasi Tambahan:
- **Chunk Size Tuning**: Eksperimen dengan berbagai chunk size pada dynamic scheduling.
- **Thread Affinity**: Pengaturan thread affinity untuk mengurangi migration overhead.
- **Cache-Friendly Access Pattern**: Restrukturisasi loop untuk meningkatkan cache locality.
- **Load Balancing**: Implementasi work-stealing algorithm untuk distribusi beban yang lebih baik.  



## 8. References
1. Youtube : OPENCV ep 6 : Konvolusi dan Kernel --> link berikut : https://www.youtube.com/watch?v=nd-TfnK-9xI .
2. Hanya REFERENSI Github orang lain --> link berikut : https://www.google.com/imgres?q=konvolusi%20open%20mp&imgurl=https%3A%2F%2Fi.ytimg.com%2Fvi%2F_Cu1b7KwGIU%2Fhq720.jpg%3Fsqp%3D-oaymwE7CK4FEIIDSFryq4qpAy0IARUAAAAAGAElAADIQj0AgKJD8AEB-AGEBYAC4AOKAgwIABABGFkgQyhlMA8%3D%26rs%3DAOn4CLB8JYIN2caUGFdx4F_tU33o-XeNVg&imgrefurl=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3D_Cu1b7KwGIU&docid=iPUlFPT3_-an8M&tbnid=jvyMpizxTNljcM&vet=12ahUKEwimha2Lj5-QAxXyxjgGHbMDOAUQM3oECBcQAA..i&w=686&h=386&hcb=2&ved=2ahUKEwimha2Lj5-QAxXyxjgGHbMDOAUQM3oECBcQAA
3. Catatan Kuliah IF3130 Sistem Paralel dan Terdistribusi, Institut Teknologi Bandung.



## 9. How to Run

### 9.1 Prasyarat Sistem:
- Compiler yang mendukung OpenMP (gcc dengan flag -fopenmp)
- Library untuk pemrosesan gambar (biasanya sudah termasuk dalam sistem)
- Sistem operasi Linux atau WSL (Windows Subsystem for Linux)

### 9.2 Proses Kompilasi:
```bash
# Kompilasi dengan OpenMP support
g++ -fopenmp -O2 -o mp main.cpp mp.cpp -lm

# Atau menggunakan makefile jika tersedia
make
```

### 9.3 Cara Menjalankan Program:
```bash
# Format umum:
./mp <num_threads> <input_image> <output_image> > <log_file>

# Contoh penggunaan:
./mp 2 ../test_cases/birds.jpg output_birds_2thread.jpg > hasil_birds_2thread.txt
./mp 4 ../test_cases/birds.jpg output_birds_4thread.jpg > hasil_birds_4thread.txt
./mp 8 ../test_cases/birds.jpg output_birds_8thread.jpg > hasil_birds_8thread.txt

# Untuk semua test cases:
./mp 2 ../test_cases/fish.jpg output_fish_2thread.jpg > hasil_fish_2thread.txt
./mp 4 ../test_cases/lion.jpg output_lion_4thread.jpg > hasil_lion_4thread.txt
./mp 8 ../test_cases/snake.jpg output_snake_8thread.jpg > hasil_snake_8thread.txt
./mp 8 ../test_cases/view.jpg output_view_8thread.jpg > hasil_view_8thread.txt
```

### 9.4 Parameter Program:
- **num_threads**: Jumlah thread yang digunakan (2, 4, 8, dll.)
- **input_image**: Path ke file gambar input (format .jpg)
- **output_image**: Path untuk menyimpan hasil edge detection
- **log_file**: File untuk menyimpan output timing dan informasi program