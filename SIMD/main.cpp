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
const int maxrow = 60000; //3000*32>90000 ,����������90000�ı���Ԫ�о���60000��
const int numBasis = 40000;   //���洢90000*40000����Ԫ��

//long long read = 0;
long long head, tail, freq;

map<int, int*>iToBasis;    //����Ϊi����Ԫ�ӵ�ӳ��
map<int, int>iToFirst;     //��i������Ԫ���Լ��������ӳ��
map<int, int*>ans;			//��

fstream RowFile("����Ԫ��.txt", ios::in | ios::out);
fstream BasisFile("��Ԫ��.txt", ios::in | ios::out);


int gRows[maxrow][maxsize];   //����Ԫ�����60000�У�3000��
int gBasis[numBasis][maxsize];  //��Ԫ�����40000�У�3000��

void reset() {
	//	read = 0;
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

void readBasis() {          //��ȡ��Ԫ��
	for (int i = 0; i < numBasis; i++) {
		if (BasisFile.eof()) {
			cout << "��ȡ��Ԫ��" << i-1 << "��" << endl;
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
				iToBasis.insert(pair<int, int*>(row, gBasis[row]));
			}
			int index = pos / 32;
			int offset = pos % 32;
			gBasis[row][index] = gBasis[row][index] | (1 << offset);
		}
		flag = false;
		row = 0;
	}
}

int readRowsFrom(int pos) {       //��ȡ����Ԫ��
	iToFirst.clear();
	if (RowFile.is_open())
		RowFile.close();
	RowFile.open("����Ԫ��.txt", ios::in | ios::out);
	memset(gRows, 0, sizeof(gRows));   //����Ϊ0
	string line;
	for (int i = 0; i < pos; i++) {       //��ȡposǰ���޹���
		getline(RowFile, line);
	}
	for (int i = pos; i < pos + maxrow; i++) {
		int tmp;
		getline(RowFile, line);
		if (line.empty()) {
			cout << "��ȡ����Ԫ�� "<<i<<" ��" << endl;
			return i;   //���ض�ȡ������
		}
		bool flag = false;
		stringstream s(line);
		while (s >> tmp) {
			if (!flag) {//i-pos���кţ�tmp������
				iToFirst.insert(pair<int, int>(i - pos, tmp));
			}
			int index = tmp / 32;
			int offset = tmp % 32;
			gRows[i - pos][index] = gRows[i - pos][index] | (1 << offset);
			flag = true;
		}
	}
	cout << "read max rows" << endl;
	return -1;  //�ɹ���ȡmaxrow��

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
	long long readBegin, readEnd;
	int begin = 0;
	int flag;
	QueryPerformanceCounter((LARGE_INTEGER*)&readBegin);
	flag = readRowsFrom(begin);     //��ȡ����Ԫ��
	QueryPerformanceCounter((LARGE_INTEGER*)&readEnd);
	head += (readEnd - readBegin);               //��ȥ��ȡ���ݵ�ʱ��

	int num = (flag == -1) ? maxrow : flag;
	for (int i = 0; i < num; i++) {
		while (iToFirst.find(i) != iToFirst.end()) {
			int first = iToFirst.find(i)->second;      //first������
			if (iToBasis.find(first) != iToBasis.end()) {  //��������Ϊfirst��Ԫ��
				int* basis = iToBasis.find(first)->second;  //�ҵ�����Ԫ�ӵ�����
				for (int j = 0; j < maxsize; j++) {
					gRows[i][j] = gRows[i][j] ^ basis[j];     //���������Ԫ

				}
				update(i);   //����map
			}
			else {   //����Ϊ��Ԫ��
				for (int j = 0; j < maxsize; j++) {
					gBasis[first][j] = gRows[i][j];
				}
				iToBasis.insert(pair<int, int*>(first, gBasis[first]));
				ans.insert(pair<int, int*>(first, gBasis[first]));
				iToFirst.erase(i);
			}
		}
	}


}

void AVX_GE() {
	long long readBegin, readEnd;
	int begin = 0;
	int flag;

	QueryPerformanceCounter((LARGE_INTEGER*)&readBegin);
	flag = readRowsFrom(begin);     //��ȡ����Ԫ��
	QueryPerformanceCounter((LARGE_INTEGER*)&readEnd);
	head += (readEnd - readBegin);              //��ȥ��ȡ���ݵ�ʱ��
	int num = (flag == -1) ? maxrow : flag;
	for (int i = 0; i < num; i++) {
		while (iToFirst.find(i) != iToFirst.end()) {
			int first = iToFirst.find(i)->second;
			if (iToBasis.find(first) != iToBasis.end()) {  //���ڸ���Ԫ��
				int* basis = iToBasis.find(first)->second;
				int j = 0;
				for (; j + 8 < maxsize; j += 8) {
					__m256i vij = _mm256_loadu_si256((__m256i*) & gRows[i][j]);
					__m256i vj = _mm256_loadu_si256((__m256i*) & basis[j]);
					__m256i vx = _mm256_xor_si256(vij, vj);
					_mm256_storeu_si256((__m256i*) & gRows[i][j], vx);
				}
				for (; j < maxsize; j++) {
					gRows[i][j] = gRows[i][j] ^ basis[j];
				}
				update(i);
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
				iToBasis.insert(pair<int, int*>(first, gBasis[first]));
				ans.insert(pair<int, int*>(first, gBasis[first]));
				iToFirst.erase(i);
			}
		}
	}


}


int main() {
	double time1 = 0;
	double time2 = 0;


	for (int i = 0; i < 1; i++) {
		ofstream out("��Ԫ���.txt");
		ofstream out1("��Ԫ���(AVX).txt");
		out << "__________" << endl;
		out1 << "__________" << endl;
		QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
		readBasis();
		//writeResult();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		GE();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		cout << "Ordinary time:" << (tail - head) * 1000 / freq << "ms" << endl;
		time1 += (tail - head) * 1000 / freq;
		writeResult(out);

		reset();

		readBasis();
		QueryPerformanceCounter((LARGE_INTEGER*)&head);
		AVX_GE();
		QueryPerformanceCounter((LARGE_INTEGER*)&tail);
		cout << "AVX time:" << (tail - head) * 1000 / freq << "ms" << endl;
		time2 += (tail - head) * 1000 / freq;
		writeResult(out1);

		reset();
		out.close();
		out1.close();
	}
	cout << "time1:" << time1 / 5 << endl << "time2:" << time2 / 5;
}
