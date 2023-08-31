//使用了ordered方式更正的OpenMP的特殊高斯消元算法
#include<iostream>
#include<fstream>
#include<string>
#include<sstream>
#include<map>
#include<windows.h>
#include<tmmintrin.h>
#include<xmmintrin.h>
#include<emmintrin.h>
#include<pmmintrin.h>
#include<smmintrin.h>
#include<nmmintrin.h>
#include<immintrin.h>
#include<pthread.h>
#include<omp.h>
using namespace std;

#define NUM_THREADS 7


struct threadParam_t {    //参数数据结构
	int t_id;
	int num;
};

const int maxsize = 3000;
const int maxrow = 60000; //3000*32>90000 ,最多存贮列数90000的被消元行矩阵60000行
const int numBasis = 100000;   //最多存储90000*100000的消元子

pthread_mutex_t lock;  //写入消元子时需要加锁

//long long read = 0;
long long head, tail, freq;

//map<int, int*>iToBasis;    //首项为i的消元子的映射
map<int, int*>ans;			//答案

fstream RowFile("被消元行.txt", ios::in | ios::out);
fstream BasisFile("消元子.txt", ios::in | ios::out);


int gRows[maxrow][maxsize];   //被消元行最多60000行，3000列
int gBasis[numBasis][maxsize];  //消元子最多40000行，3000列

int ifBasis[numBasis] = { 0 };

void reset() {
	//	read = 0;
	memset(gRows, 0, sizeof(gRows));
	memset(gBasis, 0, sizeof(gBasis));
	memset(ifBasis, 0, sizeof(ifBasis));
	RowFile.close();
	BasisFile.close();
	RowFile.open("被消元行.txt", ios::in | ios::out);
	BasisFile.open("消元子.txt", ios::in | ios::out);
	//iToBasis.clear();

	ans.clear();
}

int readBasis() {          //读取消元子
	for (int i = 0; i < numBasis; i++) {
		if (BasisFile.eof()) {
			cout << "读取消元子" << i - 1 << "行" << endl;
			return i - 1;
		}
		string tmp;
		bool flag = false;
		int row = 0;
		getline(BasisFile, tmp);
		stringstream s(tmp);
		int pos;
		while (s >> pos) {
			//cout << pos << " ";
			if (!flag) {
				row = pos;
				flag = true;
				//iToBasis.insert(pair<int, int*>(row, gBasis[row]));
				ifBasis[row] = 1;
			}
			int index = pos / 32;
			int offset = pos % 32;
			gBasis[row][index] = gBasis[row][index] | (1 << offset);
		}
		flag = false;
		row = 0;
	}
}

int readRowsFrom(int pos) {       //读取被消元行
	if (RowFile.is_open())
		RowFile.close();
	RowFile.open("被消元行.txt", ios::in | ios::out);
	memset(gRows, 0, sizeof(gRows));   //重置为0
	string line;
	for (int i = 0; i < pos; i++) {       //读取pos前的无关行
		getline(RowFile, line);
	}
	for (int i = pos; i < pos + maxrow; i++) {
		int tmp;
		getline(RowFile, line);
		if (line.empty()) {
			cout << "读取被消元行 " << i << " 行" << endl;
			return i;   //返回读取的行数
		}
		bool flag = false;
		stringstream s(line);
		while (s >> tmp) {
			int index = tmp / 32;
			int offset = tmp % 32;
			gRows[i - pos][index] = gRows[i - pos][index] | (1 << offset);
			flag = true;
		}
	}
	cout << "read max rows" << endl;
	return -1;  //成功读取maxrow行

}

int findfirst(int row) {  //寻找第row行被消元行的首项
	int first;
	for (int i = maxsize - 1; i >= 0; i--) {
		if (gRows[row][i] == 0)
			continue;
		else {
			int pos = i * 32;
			int offset = 0;
			for (int k = 31; k >= 0; k--) {
				if (gRows[row][i] & (1 << k))
				{
					offset = k;
					break;
				}
			}
			first = pos + offset;
			return first;
		}
	}
	return -1;
}



void writeResult(ofstream& out) {
	for (auto it = ans.rbegin(); it != ans.rend(); it++) {
		int* result = it->second;
		int max = it->first / 32 + 1;
		for (int i = max; i >= 0; i--) {
			if (result[i] == 0)
				continue;
			int pos = i * 32;
			//int offset = 0;
			for (int k = 31; k >= 0; k--) {
				if (result[i] & (1 << k)) {
					out << k + pos << " ";
				}
			}
		}
		out << endl;
	}
}

void GE() {
	int begin = 0;
	int flag;
	flag = readRowsFrom(begin);     //读取被消元行

	int num = (flag == -1) ? maxrow : flag;
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	for (int i = 0; i < num; i++) {
		while (findfirst(i)!= -1) {     //存在首项
			int first =findfirst(i);      //first是首项
			if (ifBasis[first]==1) {  //存在首项为first消元子
				//int* basis = iToBasis.find(first)->second;  //找到该消元子的数组
				for (int j = 0; j < maxsize; j++) {
					gRows[i][j] = gRows[i][j] ^ gBasis[first][j];     //进行异或消元

				}
			}
			else {   //升级为消元子
				//cout << first << endl;
				for (int j = 0; j < maxsize; j++) {
					gBasis[first][j] = gRows[i][j];
				}
				//iToBasis.insert(pair<int, int*>(first, gBasis[first]));
				ifBasis[first] = 1;
				ans.insert(pair<int, int*>(first, gBasis[first]));
				break;
			}
		}
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "Ordinary time:" << (tail - head) * 1000 / freq << "ms" << endl;
}

void GE_omp() {
	int begin = 0;
	int flag;
	flag = readRowsFrom(begin);     //读取被消元行
	//int i = 0, j = 0;
	int t_id = omp_get_thread_num();
	int num = (flag == -1) ? maxrow : flag;
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
#pragma omp parallel num_threads(NUM_THREADS)
	{
#pragma omp for schedule(guided)
	for (int i = 0; i < num; i++) {
		//cout << omp_get_thread_num() << "线程" << endl;
		while (findfirst(i) != -1) {     //存在首项
			int first = findfirst(i);      //first是首项
			if (ifBasis[first] == 1) {  //存在首项为first消元子
				//cout << first << "from" << omp_get_thread_num() << endl;
				for (int j = 0; j < maxsize; j++) {
					gRows[i][j] = gRows[i][j] ^ gBasis[first][j];     //进行异或消元

				}
			}
			else {   //升级为消元子
				//cout << first <<"from"<< omp_get_thread_num()<< endl;
#pragma omp critical
				if (ifBasis[first] == 0) {
					for (int j = 0; j < maxsize; j++) {
						gBasis[first][j] = gRows[i][j];
					}
					//iToBasis.insert(pair<int, int*>(first, gBasis[first]));
					ifBasis[first] = 1;
					ans.insert(pair<int, int*>(first, gBasis[first]));
				}
				//break;    //此处千万不可用break，否则会导致冲突的被消元行不被继续消元
			}

		}
	}
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "Omp time:" << (tail - head) * 1000 / freq << "ms" << endl;
}


void AVX_GE() {
	int begin = 0;
	int flag;
	flag = readRowsFrom(begin);     //读取被消元行
	int num = (flag == -1) ? maxrow : flag;
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
	for (int i = 0; i < num; i++) {
		while (findfirst(i) != -1) {
			int first = findfirst(i);
			if (ifBasis[first]==1) {  //存在该消元子
				//int* basis = iToBasis.find(first)->second;
				int j = 0;
				for (; j + 8 < maxsize; j += 8) {
					__m256i vij = _mm256_loadu_si256((__m256i*) & gRows[i][j]);
					__m256i vj = _mm256_loadu_si256((__m256i*) & gBasis[first][j]);
					__m256i vx = _mm256_xor_si256(vij, vj);
					_mm256_storeu_si256((__m256i*) & gRows[i][j], vx);
				}
				for (; j < maxsize; j++) {
					gRows[i][j] = gRows[i][j] ^ gBasis[first][j];
				}
			}
			else {
				int j = 0;
				for (; j + 8 < maxsize; j += 8) {
					__m256i vij = _mm256_loadu_si256((__m256i*) & gRows[i][j]);
					_mm256_storeu_si256((__m256i*) & gBasis[first][j], vij);
				}
				for (; j < maxsize; j++) {
					gBasis[first][j] = gRows[i][j];
				}
				//iToBasis.insert(pair<int, int*>(first, gBasis[first]));
				ifBasis[first] = 1;
				ans.insert(pair<int, int*>(first, gBasis[first]));
				break;

			}
		}
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "AVX time:" << (tail - head) * 1000 / freq << "ms" << endl;
}

void AVX_GE_omp() {
	int begin = 0;
	int flag;
	flag = readRowsFrom(begin);     //读取被消元行
	int num = (flag == -1) ? maxrow : flag;
	int i = 0, j = 0;
	QueryPerformanceCounter((LARGE_INTEGER*)&head);
#pragma omp parallel  num_threads(NUM_THREADS),private(i,j)
#pragma omp for ordered schedule(guided)
	for (i = 0; i < num; i++) {
		while (findfirst(i) != -1) {
			int first = findfirst(i);
			if (ifBasis[first]==1) {  //存在该消元子
				//int* basis = iToBasis.find(first)->second;
				j = 0;
				for (; j + 8 < maxsize; j += 8) {
					__m256i vij = _mm256_loadu_si256((__m256i*) & gRows[i][j]);
					__m256i vj = _mm256_loadu_si256((__m256i*) & gBasis[first][j]);
					__m256i vx = _mm256_xor_si256(vij, vj);
					_mm256_storeu_si256((__m256i*) & gRows[i][j], vx);
				}
				for (; j < maxsize; j++) {
					gRows[i][j] = gRows[i][j] ^ gBasis[first][j];
				}
			}
			else {
#pragma omp ordered
				{j = 0;
				for (; j + 8 < maxsize; j += 8) {
					__m256i vij = _mm256_loadu_si256((__m256i*) & gRows[i][j]);
					_mm256_storeu_si256((__m256i*) & gBasis[first][j], vij);
				}
				for (; j < maxsize; j++) {
					gBasis[first][j] = gRows[i][j];
				}
				//iToBasis.insert(pair<int, int*>(first, gBasis[first]));
				ifBasis[first] = 1;
				ans.insert(pair<int, int*>(first, gBasis[first]));
				}
			}
		}
	}
	QueryPerformanceCounter((LARGE_INTEGER*)&tail);
	cout << "AVX_omp time:" << (tail - head) * 1000 / freq << "ms" << endl;
}

int main() {

		ofstream out("消元结果.txt");
		ofstream out1("消元结果(AVX).txt");
		ofstream out2("消元结果(GE_lock).txt");
		ofstream out3("消元结果(AVX_lock).txt");
		ofstream out4("消元结果(GE_omp).txt");
		ofstream out5("消元结果(AVX_omp).txt");
		QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

		readBasis();
		GE();
		writeResult(out);

		reset();

		readBasis();
		GE_omp();
		writeResult(out4);

		reset();

		readBasis();
		AVX_GE_omp();
		writeResult(out5);

		reset();

		readBasis();
		AVX_GE();
		writeResult(out1);

		reset();

		out.close();
		out1.close();
		out2.close();
		out3.close();
}
