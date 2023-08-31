//本文件中包含了:
//普通MPI、非阻塞MPI
//流水线版本优化的MPI Lu_mpi_pipeline
//且均能正常运行

#include<iostream>
#include <stdio.h>
#include<cstring>
#include<typeinfo>
#include <stdlib.h>
#include<cmath>
#include<mpi.h>
#include<windows.h>
#include<omp.h>
#include<tmmintrin.h>
#include<xmmintrin.h>
#include<emmintrin.h>
#include<pmmintrin.h>
#include<smmintrin.h>
#include<nmmintrin.h>
#include<immintrin.h>

using namespace std;
#define N 500
#define NUM_THREADS 7
float** A = NULL;

long long head, tail, freq;

void A_init() {     //未对齐的数组的初始化
    A = new float* [N];
    for (int i = 0; i < N; i++) {
        A[i] = new float[N];
    }
    for (int i = 0; i < N; i++) {
        A[i][i] = 1.0;
        for (int j = i + 1; j < N; j++) {
            A[i][j] = rand() % 5000;
        }

    }
    for (int k = 0; k < N; k++) {
        for (int i = k + 1; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A[i][j] += A[k][j];
                A[i][j] = (int)A[i][j] % 5000;
            }
        }
    }
}
void A_initAsEmpty() {
    A = new float* [N];
    for (int i = 0; i < N; i++) {
        A[i] = new float[N];
        memset(A[i], 0, N * sizeof(float));
    }

}

void deleteA() {
    for (int i = 0; i < N; i++) {
        delete[] A[i];
    }
    delete A;
}

void print(float** a) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            cout << a[i][j] << " ";
        }
        cout << endl;
    }
}

void LU() {    //普通消元算法
    for (int k = 0; k < N; k++) {
        for (int j = k + 1; j < N; j++) {
            A[k][j] = A[k][j] / A[k][k];
        }
        A[k][k] = 1.0;

        for (int i = k + 1; i < N; i++) {
            for (int j = k + 1; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
}


double LU_mpi(int argc, char* argv[]) {  //块划分
    double start_time = 0;
    double end_time = 0;
    MPI_Init(&argc, &argv);
    int total = 0;
    int rank = 0;
    int i = 0;
    int j = 0;
    int k = 0;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = N / total * rank;
    int end = (rank == total - 1) ? N : N / total * (rank + 1);
    if (rank == 0) {  //0号进程初始化矩阵
        A_init();

        for (j = 1; j < total; j++) {
            int b = j * (N / total), e = (j == total - 1) ? N : (j + 1) * (N / total);
            for (i = b; i < e; i++) {
                MPI_Send(&A[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD);//1是初始矩阵信息，向每个进程发送数据
            }
        }

    }
    else {
        A_initAsEmpty();
        for (i = begin; i < end; i++) {
            MPI_Recv(&A[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
        }

    }

    MPI_Barrier(MPI_COMM_WORLD);  //此时每个进程都拿到了数据
    start_time = MPI_Wtime();
    for (k = 0; k < N; k++) {
        if ((begin <= k && k < end)) {
            for (j = k + 1; j < N; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            for (j = 0; j < total; j++) { //
                if (j != rank)
                    MPI_Send(&A[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD);//0号消息表示除法完毕
            }
        }
        else {
            int src;
            if (k < N / total * total)//在可均分的任务量内
                src = k / (N / total);
            else
                src = total - 1;
            MPI_Recv(&A[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &status);
        }
        for (i = max(begin, k + 1); i < end; i++) {
            for (j = k + 1; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);	//各进程同步
    if (rank == 0) {//0号进程中存有最终结果
        end_time = MPI_Wtime();
        printf("平凡MPI，块划分耗时：%.4lf ms\n", 1000 * (end_time - start_time));
    }
    MPI_Finalize();
    return end_time - start_time;
}

double LU_mpi_async(int argc, char* argv[]) {  //非阻塞通信
    double start_time = 0;
    double end_time = 0;
    MPI_Init(&argc, &argv);
    int total = 0;
    int rank = 0;
    int i = 0;
    int j = 0;
    int k = 0;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = N / total * rank;
    int end = (rank == total - 1) ? N : N / total * (rank + 1);

    if (rank == 0) {  //0号进程初始化矩阵
        A_init();
        MPI_Request* request = new MPI_Request[N - end];
        for (j = 1; j < total; j++) {
            int b = j * (N / total), e = (j == total - 1) ? N : (j + 1) * (N / total);

            for (i = b; i < e; i++) {
                MPI_Isend(&A[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD, &request[i - end]);//非阻塞传递矩阵数据
            }

        }
        MPI_Waitall(N - end, request, MPI_STATUS_IGNORE); //等待传递

    }
    else {
        A_initAsEmpty();
        MPI_Request* request = new MPI_Request[end - begin];
        for (i = begin; i < end; i++) {
            MPI_Irecv(&A[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &request[i - begin]);  //非阻塞接收
        }
        MPI_Waitall(end - begin, request, MPI_STATUS_IGNORE);

    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    for (k = 0; k < N; k++) {
        if ((begin <= k && k < end)) {
            for (j = k + 1; j < N; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            MPI_Request* request = new MPI_Request[total - 1 - rank];  //非阻塞传递
            for (j = rank + 1; j < total; j++) { //块划分中，已经消元好且进行了除法置1的行向量仅

                MPI_Isend(&A[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request[j - rank - 1]);//0号消息表示除法完毕
            }
            MPI_Waitall(total - 1 - rank, request, MPI_STATUS_IGNORE);
            if (k == end - 1)
                break; //若执行完自身的任务，可直接跳出
        }
        else {
            int src = k / (N / total);
            MPI_Request request;
            MPI_Irecv(&A[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, MPI_STATUS_IGNORE);         //实际上仍然是阻塞接收，因为接下来的操作需要这些数据
        }
        for (i = max(begin, k + 1); i < end; i++) {
            for (j = k + 1; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);	//各进程同步
    if (rank == total - 1) {
        end_time = MPI_Wtime();
        printf("平凡MPI，块划分+非阻塞耗时：%.4lf ms\n", 1000 * (end_time - start_time));
        //print(A);
    }
    MPI_Finalize();
    return end_time - start_time;
}


double LU_mpi_pipeline(int argc, char* argv[]) {  //稍做优化的块划分
    double start_time = 0;
    double end_time = 0;
    MPI_Init(&argc, &argv);
    int total = 0;
    int rank = 0;
    int i = 0;
    int j = 0;
    int k = 0;
    MPI_Status status;
    MPI_Comm_size(MPI_COMM_WORLD, &total);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int begin = N / total * rank;
    int end = (rank == total - 1) ? N : N / total * (rank + 1);
    if (rank == 0) {  //0号进程初始化矩阵
        A_init();

        for (j = 1; j < total; j++) {
            int b = j * (N / total), e = (j == total - 1) ? N : (j + 1) * (N / total);
            for (i = b; i < e; i++) {
                MPI_Send(&A[i][0], N, MPI_FLOAT, j, 1, MPI_COMM_WORLD);//1是初始矩阵信息，向每个进程发送数据
            }
        }

    }
    else {
        A_initAsEmpty();
        for (i = begin; i < end; i++) {
            MPI_Recv(&A[i][0], N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
        }

    }

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    for (k = 0; k < N; k++) {
        if ((begin <= k && k < end)) {
            for (j = k + 1; j < N; j++) {
                A[k][j] = A[k][j] / A[k][k];
            }
            A[k][k] = 1.0;
            MPI_Request* request = new MPI_Request[total - 1 - rank];  //非阻塞传递
            for (j = rank + 1; j < total; j++) { //块划分中，已经消元好且进行了除法置1的行向量仅

                MPI_Isend(&A[k][0], N, MPI_FLOAT, j, 0, MPI_COMM_WORLD, &request[j - rank - 1]);//0号消息表示除法完毕
            }
            MPI_Waitall(total - 1 - rank, request, MPI_STATUS_IGNORE);
            if (k == end - 1)
                break; //若执行完自身的任务，可直接跳出
        }
        else {
            int src = k / (N / total);
            MPI_Request request;
            MPI_Irecv(&A[k][0], N, MPI_FLOAT, src, 0, MPI_COMM_WORLD, &request);
            MPI_Wait(&request, MPI_STATUS_IGNORE);         //实际上仍然是阻塞接收，因为接下来的操作需要这些数据
        }
        for (i = max(begin, k + 1); i < end; i++) {
            for (j = k + 1; j < N; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            A[i][k] = 0;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);	//各进程同步
    if (rank == total - 1) {
        end_time = MPI_Wtime();
        printf("平凡MPI，流水线优化耗时：%.4lf ms\n", 1000 * (end_time - start_time));
        //print(A);
    }
    MPI_Finalize();
    return end_time - start_time;
}

void cal(void(*func)()) {
    A_init();
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    func();
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);

}

int main(int argc, char* argv[]) {
    /*   QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
       cal(LU);
       cout << "平凡算法耗时：" << (tail - head) * 1000 / freq << "ms" << endl;
       deleteA();*/

    LU_mpi_pipeline(argc, argv);



}

