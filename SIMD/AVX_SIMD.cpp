#include<iostream>
#include<windows.h>
#include <stdio.h>
#include<typeinfo>
#include <stdlib.h>
#include<tmmintrin.h>
#include<xmmintrin.h>
#include<emmintrin.h>
#include<pmmintrin.h>
#include<smmintrin.h>
#include<nmmintrin.h>
#include<immintrin.h>
using namespace std;
#define N 2000
float** align_m = NULL;  //对齐的数组
float** unalign = NULL;  // 未对齐的数组

void unalign_init() {     //未对齐的数组的初始化
    unalign = new float*[N];
    for (int i = 0; i < N; i++) {
        unalign[i] = new float[N];
    }
    for (int i = 0; i < N; i++) {
        unalign[i][i] = 1.0;
        for (int j = i + 1; j < N; j++) {
            unalign[i][j] = rand() % 1000;
        }

    }
    for (int k = 0; k < N; k++) {
        for (int i = k + 1; i < N; i++) {
            for (int j = 0; j < N; j++) {
                unalign[i][j] += unalign[k][j];
                unalign[i][j] = (int)unalign[i][j] % 1000;
            }
        }
    }
}

void deleteUnalign() {
    for (int i = 0; i < N; i++) {
        delete[] unalign[i];
    }
    delete unalign;
}

void align_init(int alignment) {
    if (align_m == NULL) {
        align_m = (float**)_aligned_malloc(sizeof(float*) * N, alignment);
        for (int i = 0; i < N; i++) {
            align_m[i] = (float*)_aligned_malloc(sizeof(float) * N, alignment);
            //使得矩阵每一行在内存中按照alignment对齐，SSE为16，AVX为32
            //cout << align_m[i] << endl;
        }
    }
    for (int i = 0; i < N; i++) {
        align_m[i][i] = 1.0;          //对齐矩阵的初始化
        for (int j = i + 1; j < N; j++) {
            align_m[i][j] = rand() % 1000;
        }

    }
    for (int k = 0; k < N; k++) {
        for (int i = k + 1; i < N; i++) {
            for (int j = 0; j < N; j++) {
                align_m[i][j] += align_m[k][j];
                align_m[i][j] = (int)align_m[i][j] % 1000;
            }
        }
    }
}


void print(float **a) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << a[i][j] << " ";
        }
        cout << endl;
    }
}

void LU(float **m) {    //普通消元算法
    for (int k = 0; k < N; k++) {
        for (int j = k + 1; j < N; j++) {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < N; i++) {
            for (int j = k + 1; j < N; j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
    }
}

void avx_optimized(float **m) {
    for (int k = 0; k < N; k++) {
        __m256 t1 = _mm256_set1_ps(m[k][k]);
        int j = 0;
        for (j = k + 1; j + 8 <= N; j += 8) {
            __m256 t2 = _mm256_loadu_ps(&m[k][j]);
            t2 = _mm256_div_ps(t2, t1);
            _mm256_storeu_ps(&m[k][j], t2);
        }
        for (; j < N; j++) {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < N; i++) {
            __m256 vik = _mm256_set1_ps(m[i][k]);
            for (j = k + 1; j + 8 <= N; j += 8) {
                __m256 vkj = _mm256_loadu_ps(&m[k][j]);
                __m256 vij = _mm256_loadu_ps(&m[i][j]);
                __m256 vx = _mm256_mul_ps(vik, vkj);
                vij = _mm256_sub_ps(vij, vx);
                _mm256_storeu_ps(&m[i][j], vij);
            }
            for (; j < N; j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
    }
}

void avx_align(float** m) {
    for (int k = 0; k < N; k++) {
        __m256 t1 = _mm256_set1_ps(m[k][k]);
        int j = k+1;
        while ((int)(&m[k][j]) % 32)
        {
            m[k][j] = m[k][j] / m[k][k];
            j++;
        }
        for ( ; j + 8 <= N; j += 8) {
            __m256 t2 = _mm256_load_ps(&m[k][j]);
            t2 = _mm256_div_ps(t2, t1);
            _mm256_store_ps(&m[k][j], t2);
        }
        for ( ; j < N; j++) {
            m[k][j] = m[k][j] / m[k][k];
        }
        m[k][k] = 1.0;
        for (int i = k + 1; i < N; i++) {
            __m256 vik = _mm256_set1_ps(m[i][k]);
            j = k + 1;
            while ((int)(&m[k][j]) % 32)
            {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
                j++;
            }
            for ( ; j + 8 <= N; j += 8) {
                __m256 vkj = _mm256_load_ps(&m[k][j]);
                __m256 vij = _mm256_loadu_ps(&m[i][j]);
                __m256 vx = _mm256_mul_ps(vik, vkj);
                vij = _mm256_sub_ps(vij, vx);
                _mm256_storeu_ps(&m[i][j], vij);
            }
            for ( ; j < N; j++) {
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            }
            m[i][k] = 0;
        }
    }
}

int main() {
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    unalign_init();
    /*for (int i = 0; i < N; i++) {
        cout << unalign[i] << endl;
    }*/
    //print(unalign);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    LU(unalign);
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "LU time:" << (tail - head) * 1000 / freq << "ms" << endl;
    cout << "------------------" << endl;
    deleteUnalign();
         
     unalign_init();
     //print(unalign);
     QueryPerformanceCounter((LARGE_INTEGER*)&head);
     avx_optimized(unalign);
     QueryPerformanceCounter((LARGE_INTEGER*)&tail);
     //print(unalign);
     cout << "avx(未对齐):" << (tail - head) * 1000 / freq << "ms" << endl;
     cout << "------------------" << endl;
     deleteUnalign();

     align_init(32);     //AVX需要32位对齐
     //print(align_m);
     QueryPerformanceCounter((LARGE_INTEGER*)&head);
     avx_align(align_m);
     QueryPerformanceCounter((LARGE_INTEGER*)&tail);
     cout << "avx(对齐) time:" << (tail - head) * 1000 / freq << "ms" << endl;
     //print(align_m);

}