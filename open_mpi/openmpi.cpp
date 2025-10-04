#include <opencv2/opencv.hpp>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>

using namespace std;
using namespace cv;

// terkait MPI fungsi 
// 1. MPI init : bakal inisialize MPI 
// 2. MPI Comm_rank : dapetin rank proses
// 3. MPI Comm_size : dapetin total proses 
// 4. MPI Bcast : broadcast data dari root ke semua proses (nyebarin data 1 proses ke semua proses)
// 5. MPI Scatter : nyebarin data array ke semua proses (misal array 100 elemen, 4 proses, tiap proses dapet 25 elemen)
// 6. MPI Gather : ngumpulin data array dari semua proses ke root (misal tiap proses punya 25 elemen, 4 proses, di root jadi 100 elemen)
// 7. MPI Send : ngirim data ke proses tertentu
// 8. MPI Recv : nerima data dari proses tertentu
// 9. MPI Reduce : ngumpulin data dari semua proses dan ngelakuin operasi (misal sum, max, min) terus hasilnya di root
// 10. MPI Finalize : nutup MPI


// Struktur untuk menyimpan hasil pengukuran waktu
struct WaktuEksekusi {
    double input;      // Waktu untuk membaca input
    double processing; // Waktu untuk pemrosesan Sobel
    double output;     // Waktu untuk menyimpan output
};

// Kernel Sobel horizontal dan vertikal
const int kernelHorizontal[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

const int kernelVertikal[3][3] = {
    {1, 2, 1},
    {0, 0, 0},
    {-1,-2,-1}
};

// Fungsi hitung konvolusi satu piksel
pair<int, int> hitungKonvolusi(const Mat& gambar, int baris, int kolom) {
    int gradienX = 0, gradienY = 0;
    
    // kernel 3x3 di sekitar piksel (baris, kolom)
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            int barisTarget = baris + i;
            int kolomTarget = kolom + j;
            
            // Pastikan koordinat masih dalam batas gambar
            if (barisTarget >= 0 && barisTarget < gambar.rows && 
                kolomTarget >= 0 && kolomTarget < gambar.cols) {
                
                uchar nilaiPiksel = gambar.at<uchar>(barisTarget, kolomTarget);
                
                // Hitung gradien dengan kernel horizontal dan vertikal
                gradienX += nilaiPiksel * kernelHorizontal[i + 1][j + 1];
                gradienY += nilaiPiksel * kernelVertikal[i + 1][j + 1];
            }
        }
    }
    
    return make_pair(gradienX, gradienY);
}

// Fungsi untuk menghitung magnitude gradien
int hitungMagnitude(int gradienX, int gradienY) {
    double magnitude = sqrt(gradienX * gradienX + gradienY * gradienY);
    return min(255, (int)magnitude); // Batasi pada rentang 0-255
}

// Fungsi untuk menerapkan mode operasi yang berbeda
uchar terapkanMode(int magnitude, int mode, const vector<int>& threshold) {
    switch (mode) {
        case 0: // Mode Gradient Magnitude
            return (uchar)magnitude;
            
        case 1: // Mode Binary Threshold - Inverted
            return (magnitude > threshold[0]) ? 0 : 255; // Logika terbalik
            
        default: // Mode Multi-Level Threshold (n >= 2)
            int jumlahBin = threshold.size() + 1;
            int levelStep = 255 / (jumlahBin - 1);
            
            // Tentukan level berdasarkan threshold
            for (int i = 0; i < threshold.size(); i++) {
                if (magnitude <= threshold[i]) {
                    return (uchar)(i * levelStep);
                }
            }
            
            // Jika magnitude lebih besar dari threshold terakhir
            return (uchar)(255);
    }
}

// Fungsi untuk memproses bagian gambar yang ditugaskan ke proses ini
Mat prosesSobelMPI(const Mat& gambar, int startBaris, int endBaris, 
                   int mode, const vector<int>& threshold) {
    Mat hasil = Mat::zeros(endBaris - startBaris, gambar.cols, CV_8UC1);
    
    for (int i = startBaris; i < endBaris; i++) {
        for (int j = 0; j < gambar.cols; j++) {
            // Hitung gradien pada posisi (i, j)
            pair<int, int> gradien = hitungKonvolusi(gambar, i, j);
            int magnitude = hitungMagnitude(gradien.first, gradien.second);
            
            // Terapkan mode operasi
            uchar nilaiOutput = terapkanMode(magnitude, mode, threshold);
            hasil.at<uchar>(i - startBaris, j) = nilaiOutput;
        }
    }
    
    return hasil;
}

int main(int argc, char** argv) {
    // Inisialisasi MPI
    MPI_Init(&argc, &argv); // inisialisasi MPI 
    
    int rank, ukuranProses;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // dapetin rank proses
    MPI_Comm_size(MPI_COMM_WORLD, &ukuranProses); // dapetin total proses
    
    // Validasi parameter command line
    if (argc < 4) {
        if (rank == 0) {
            cout << "Penggunaan: " << argv[0] << " <mode> <input.jpg> <output.jpg> [threshold...]" << endl;
            cout << "Mode 0: Gradient Magnitude" << endl;
            cout << "Mode 1: Binary Threshold (threshold diperlukan)" << endl;
            cout << "Mode n: Multi-Level Threshold (n threshold diperlukan)" << endl;
        }
        MPI_Finalize();
        return -1;
    }
    
    // Parse parameter
    int mode = atoi(argv[1]);
    string namaFileInput = argv[2];
    string namaFileOutput = argv[3];
    
    vector<int> threshold;
    for (int i = 4; i < argc; i++) {
        threshold.push_back(atoi(argv[i]));
    }
    
    // Validasi threshold berdasarkan mode
    if (mode == 1 && threshold.size() != 1) {
        if (rank == 0) {
            cout << "Mode 1 memerlukan tepat 1 threshold" << endl;
        }
        MPI_Finalize();
        return -1;
    } else if (mode >= 2 && threshold.size() != mode) {
        if (rank == 0) {
            cout << "Mode " << mode << " memerlukan " << mode << " threshold" << endl;
        }
        MPI_Finalize();
        return -1;
    }
    
    WaktuEksekusi waktu = {0.0, 0.0, 0.0};
    Mat gambarInput, hasilAkhir;
    int tinggiGambar = 0, lebarGambar = 0;
    
    // Proses root membaca input
    if (rank == 0) {
        auto mulaiInput = chrono::high_resolution_clock::now();
        
        gambarInput = imread(namaFileInput, IMREAD_GRAYSCALE);
        if (gambarInput.empty()) {
            cout << "Error,ga bisa baca file" << namaFileInput << endl;
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        
        tinggiGambar = gambarInput.rows;
        lebarGambar = gambarInput.cols;
        
        auto selesaiInput = chrono::high_resolution_clock::now();
        waktu.input = chrono::duration<double>(selesaiInput - mulaiInput).count();
        
        cout << "Gambar berhasil dimuat: " << tinggiGambar << "x" << lebarGambar << endl;
        cout << "Mode operasi: " << mode << endl;
    }
    
    // Broadcast dimensi gambar ke semua proses
    MPI_Bcast(&tinggiGambar, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&lebarGambar, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Alokasi memori untuk gambar di semua proses
    if (rank != 0) {
        gambarInput = Mat::zeros(tinggiGambar, lebarGambar, CV_8UC1);
    }
    
    // Broadcast data gambar ke semua proses
    MPI_Bcast(gambarInput.data, tinggiGambar * lebarGambar, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    
    // Hitung pembagian kerja untuk setiap proses
    int barisPerProses = tinggiGambar / ukuranProses;
    int sisaBaris = tinggiGambar % ukuranProses;
    
    int startBaris = rank * barisPerProses + min(rank, sisaBaris);
    int jumlahBaris = barisPerProses + (rank < sisaBaris ? 1 : 0);
    int endBaris = startBaris + jumlahBaris;
    
    if (rank == 0) {
        cout << "Pembagian kerja:" << endl;
        for (int i = 0; i < ukuranProses; i++) {
            int start = i * barisPerProses + min(i, sisaBaris);
            int jumlah = barisPerProses + (i < sisaBaris ? 1 : 0);
            cout << "Proses " << i << ": baris " << start << " - " << (start + jumlah - 1) << endl;
        }
    }
    
    // Mulai pemrosesan Sobel
    auto mulaiProcessing = chrono::high_resolution_clock::now();
    
    Mat hasilLokal = prosesSobelMPI(gambarInput, startBaris, endBaris, mode, threshold);
    
    auto selesaiProcessing = chrono::high_resolution_clock::now();
    double waktuProcessingLokal = chrono::duration<double>(selesaiProcessing - mulaiProcessing).count();
    
    // Kumpulkan waktu processing dari semua proses
    double waktuProcessingMaksimal;
    MPI_Reduce(&waktuProcessingLokal, &waktuProcessingMaksimal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        waktu.processing = waktuProcessingMaksimal;
    }
    
    // Kumpulkan hasil dari semua proses
    if (rank == 0) {
        hasilAkhir = Mat::zeros(tinggiGambar, lebarGambar, CV_8UC1);
        
        // Salin hasil dari proses root
        hasilLokal.copyTo(hasilAkhir(Rect(0, startBaris, lebarGambar, jumlahBaris)));
        
        // Terima hasil dari proses lain
        for (int i = 1; i < ukuranProses; i++) {
            int startBarisProses = i * barisPerProses + min(i, sisaBaris);
            int jumlahBarisProses = barisPerProses + (i < sisaBaris ? 1 : 0);
            
            Mat bufferTerima(jumlahBarisProses, lebarGambar, CV_8UC1);
            MPI_Recv(bufferTerima.data, jumlahBarisProses * lebarGambar, 
                    MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            bufferTerima.copyTo(hasilAkhir(Rect(0, startBarisProses, lebarGambar, jumlahBarisProses)));
        }
    } else {
        // Kirim hasil ke proses root
        MPI_Send(hasilLokal.data, jumlahBaris * lebarGambar, 
                MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
    }
    
    // Proses root menyimpan output
    if (rank == 0) {
        auto mulaiOutput = chrono::high_resolution_clock::now();
        
        bool berhasilSimpan = imwrite(namaFileOutput, hasilAkhir);
        if (!berhasilSimpan) {
            cout << "Error: Tidak dapat menyimpan file " << namaFileOutput << endl;
        } else {
            cout << "Hasil berhasil disimpan ke " << namaFileOutput << endl;
        }
        
        auto selesaiOutput = chrono::high_resolution_clock::now();
        waktu.output = chrono::duration<double>(selesaiOutput - mulaiOutput).count();
        
        // Cetak laporan ringkasan
        cout << "\n=== LAPORAN RINGKASAN ===" << endl;
        cout << "File Input: " << namaFileInput << endl;
        cout << "File Output: " << namaFileOutput << endl;
        cout << "Dimensi Gambar: " << tinggiGambar << "x" << lebarGambar << endl;
        cout << "Mode Operasi: " << mode << endl;
        cout << "Jumlah Proses MPI: " << ukuranProses << endl;
        
        if (!threshold.empty()) {
            cout << "Threshold: ";
            for (int i = 0; i < threshold.size(); i++) {
                cout << threshold[i];
                if (i < threshold.size() - 1) cout << ", ";
            }
            cout << endl;
        }
        
        cout << "\n=== WAKTU EKSEKUSI ===" << endl;
        cout << "Input Time: " << waktu.input << " detik" << endl;
        cout << "Processing Time: " << waktu.processing << " detik" << endl;
        cout << "Output Time: " << waktu.output << " detik" << endl;
        cout << "Total Time: " << (waktu.input + waktu.processing + waktu.output) << " detik" << endl;
        cout << "========================" << endl;
    }
    
    MPI_Finalize();
    return 0;
}