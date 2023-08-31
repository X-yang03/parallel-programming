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
using namespace std;

const int maxsize = 3000;
const int maxrow = 3000; //3000*32>90000 ,������90000*90000�ľ���3000��
const int numBasis = 90000;

long long head, tail, freq;

map<int, int*>iToBasis;
map<int, int>iToFirst;
map<int, int*>ans;

fstream RowFile("����Ԫ��.txt", ios::in | ios::out);
fstream BasisFile("��Ԫ��.txt", ios::in | ios::out);




int gRows[maxrow][maxsize];
int gBasis[numBasis][maxsize];

void reset() {
	memset(gRows, 0, sizeof(gRows));
	memset(gBasis, 0, sizeof(gBasis));
	RowFile.close();
	BasisFile.close();
	RowFile.open("����Ԫ��.txt", ios::in | ios::out);
	BasisFile.open("��Ԫ��.txt", ios::in | ios::out);
	iToBasis.clear();
	iToFirst.clear();
	ans.clear();

}

void readBasis() {
	for (int i = 0; i < maxrow; i++) {
		if (BasisFile.eof()) {
			return;
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
				iToBasis.insert(pair<int,int*>(row, gBasis[row]));
			}
			int index = pos / 32;
			int offset = pos % 32;
			gBasis[row][index] = gBasis[row][index]| (1 << offset);
		}
		flag = false;
		row = 0;
	}
}

int readRowsFrom(int pos) {
	iToFirst.clear();
	if (RowFile.is_open())
		RowFile.close();
	RowFile.open("����Ԫ��.txt", ios::in | ios::out);
	memset(gRows, 0, sizeof(gRows));   //����Ϊ0
	string line;
	for (int i = 0; i < pos; i++) {
		getline(RowFile, line);
	}
	for (int i = pos; i < pos + maxsize; i++) {
		int tmp;
		getline(RowFile, line);
		if (line.empty()) {
			cout << "End of File!" << endl;
			return i;   //û�ܶ�ȡ��3000�У����ض�ȡ������
		}
		bool flag = false;
		stringstream s(line);
		while (s >> tmp) {
			if (!flag) {//i-pos���кţ�tmp������
				iToFirst.insert(pair<int, int>(i - pos, tmp));
			}
			int index = tmp / 32;
			int offset = tmp % 32;
			gRows[i-pos][index] = gRows[i-pos][index] | (1 << offset);
			flag = true;
		}
	}
	return -1;  //�ɹ���ȡ3000��

}

void update(int row) {
	bool flag = 0;
	for (int i = maxsize - 1; i >= 0; i--) {
		if (gRows[row][i] == 0)
			continue;
		else {
			if (!flag)
				flag = true;
			int pos = i * 32;
			int offset = 0;
			for (int k = 31; k >= 0; k--) {
				if (gRows[row][i] & (1 << k))
				{
					offset = k;
					break;
				}
			}
			int newfirst = pos + offset;
			iToFirst.erase(row);
			iToFirst.insert(pair<int, int>(row, newfirst));
			break;
		}
	}
	if (!flag) {
		iToFirst.erase(row);
	}
	return;
}

void writeResult(ofstream& out) {
	for (auto it = ans.rbegin(); it != ans.rend(); it++) {
		int* result = it->second;
		int max = it->first/32 + 1;
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
	while (true) {
		flag = readRowsFrom(begin);
		QueryPerformanceFrequency((LARGE_INTEGER*)&head);
		int num = (flag == -1) ? maxsize : flag;
		for (int i = 0; i < num; i++) {
			while (iToFirst.find(i) != iToFirst.end()) {
				int first = iToFirst.find(i)->second;
				if (iToBasis.find(first) != iToBasis.end()) {  //���ڸ���Ԫ��
					int* basis = iToBasis.find(first)->second;
					for (int j = 0; j < maxsize; j++) {
						gRows[i][j] = gRows[i][j] ^ basis[j];

					}
					update(i);
				}
		 		else {
					for (int j = 0; j < maxsize; j++) {
						gBasis[first][j] = gRows[i][j];
					}
					iToBasis.insert(pair<int,int*>(first, gBasis[first]));
					ans.insert(pair<int, int*>(first, gBasis[first]));
					iToFirst.erase(i);
				}
			}
			QueryPerformanceFrequency((LARGE_INTEGER*)&tail);
		}
		if (flag == -1)
			begin += maxsize;
		else
			break;
	}

}

void AVX_GE() {
	int begin = 0;
	int flag;
	
	while (true) {
		flag = readRowsFrom(begin);
		QueryPerformanceFrequency((LARGE_INTEGER*)&head);
		int num = (flag == -1) ? maxsize : flag;
		for (int i = 0; i < num; i++) {
			while (iToFirst.find(i) != iToFirst.end()) {
				int first = iToFirst.find(i)->second;
				if (iToBasis.find(first) != iToBasis.end()) {  //���ڸ���Ԫ��
					int* basis = iToBasis.find(first)->second;
					int j = 0;
					for ( ; j+8 < maxsize; j+=8) {
						__m256i vij = _mm256_loadu_si256((__m256i*)&gRows[i][j]);
						__m256i vj = _mm256_loadu_si256((__m256i*) & basis[j]);
						__m256i vx = _mm256_xor_si256(vij, vj);
						_mm256_storeu_si256((__m256i*) & gRows[i][j],vx);
					}
					for (; j < maxsize; j++) {
						gRows[i][j] = gRows[i][j] ^ basis[j];
					}
					update(i);
				}
				else {
					int j=0;
					for ( ; j+8 < maxsize; j+=8) {
						__m256i vij = _mm256_loadu_si256((__m256i*)&gRows[i][j]);
						_mm256_storeu_si256((__m256i*) & gRows[first][j],vij);
					}
					for ( ; j < maxsize; j++) {
						gBasis[first][j] = gRows[i][j];
					}
					iToBasis.insert(pair<int, int*>(first, gBasis[first]));
					ans.insert(pair<int, int*>(first, gBasis[first]));
					iToFirst.erase(i);
				}
			}
		QueryPerformanceFrequency((LARGE_INTEGER*)&tail);
		}
		if (flag == -1)
			begin += maxsize;
		else
			break;
	}
}

void SSE_GE() {
	int begin = 0;
	int flag;
	
	while (true) {
		flag = readRowsFrom(begin);
		QueryPerformanceFrequency((LARGE_INTEGER*)&head);
		int num = (flag == -1) ? maxsize : flag;
		for (int i = 0; i < num; i++) {
			while (iToFirst.find(i) != iToFirst.end()) {
				int first = iToFirst.find(i)->second;
				if (iToBasis.find(first) != iToBasis.end()) {  //���ڸ���Ԫ��
					int* basis = iToBasis.find(first)->second;
					int j = 0;
					for ( ; j+4 < maxsize; j+=4) {
						__m128i vij = _mm_loadu_si128((__m128i*)&gRows[i][j]);
						__m128i vj = _mm_loadu_si128((__m128i*) & basis[j]);
						__m128i vx = _mm_xor_si128(vij, vj);
						_mm_storeu_si128((__m128i*) & gRows[i][j],vx);
					}
					for (; j < maxsize; j++) {
						gRows[i][j] = gRows[i][j] ^ basis[j];
					}
					update(i);
				}
				else {
					int j=0;
					for ( ; j+4 < maxsize; j+=4) {
						__m128i vij = _mm_loadu_si128((__m128i*)&gRows[i][j]);
						_mm_storeu_si128((__m128i*) & gRows[first][j],vij);
					}
					for ( ; j < maxsize; j++) {
						gBasis[first][j] = gRows[i][j];
					}
					iToBasis.insert(pair<int, int*>(first, gBasis[first]));
					ans.insert(pair<int, int*>(first, gBasis[first]));
					iToFirst.erase(i);
				}
			}
		QueryPerformanceFrequency((LARGE_INTEGER*)&tail);
		}
		if (flag == -1)
			begin += maxsize;
		else
			break;
	}
}

int main() {
	double time1 = 0;
	double time2 = 0;

	
	for (int i = 0; i < 5; i++) {
        ofstream out("��Ԫ���.txt");
	    ofstream out1("��Ԫ���(AVX).txt");
        out<<"__________"<<endl;
        out1<<"__________"<<endl;
		QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
	    readBasis();
		//writeResult();
		GE();
		cout << "Ordinary time:" << (tail - head) * 1000 / freq << "ms" << endl;
		time1 += (tail - head) * 1000 / freq;
		writeResult(out);

		reset();

		readBasis();
		
		AVX_GE();
		
		//cout << "AVX time:" << (tail - head) * 1000 / freq << "ms" << endl;
		time2 += (tail - head) * 1000 / freq;
		writeResult(out1);

		reset();
		out.close();
		out1.close();
	}
	cout << "time1:" << time1/5  << endl << "time2:" << time2/5 ;
}
