#include <iostream>
#include <fstream>
#include <chrono>
#include <ratio>
#include <iomanip>
#include <random>
#include <thread>
#include <functional>

#include <CL/sycl.hpp>

using namespace cl::sycl;

long long freq, head, tail;

void print(buffer<float, 2>& buf) {
	host_accessor m{ buf ,read_only };
	auto range = m.get_range();
	for (int i = 0; i < range[0]; i++) {
		for (int j = 0; j < range[1]; j++) {
			std::cout << std::setw(16) << m[i][j];
		}
		std::cout << std::endl;
	}
}

void matrix_copy(buffer<float, 2>& to, buffer<float, 2>& from) {
	host_accessor src{ from ,read_only };
	host_accessor des{ to ,write_only };
	assert(src.get_range() == des.get_range());
	auto range = src.get_range();
	for (int i = 0; i < range[0]; i++) {
		for (int j = 0; j < range[1]; j++) {
			des[i][j] = src[i][j];
		}
	}
}

void matrix_init(buffer<float, 2>& buf) {
	host_accessor A{ buf ,read_write };

	static std::default_random_engine generator(1337);
	static std::uniform_real_distribution<float> distribution(-1.0, 1.0);

	int N = A.get_range()[0];
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

void LU(buffer<float, 2>& buf, queue& q) {
	host_accessor m{ buf ,read_write };
	int n = m.get_range()[0];
	for (int k = 0; k < n; k++) {
		for (int j = k + 1; j < n; j++) {
			m[k][j] = m[k][j] / m[k][k];
		}
		m[k][k] = 1;
		for (int i = k + 1; i < n; i++) {
			for (int j = k + 1; j < n; j++) {
				m[i][j] = m[i][j] - m[i][k] * m[k][j];
			}
			m[i][k] = 0;
		}
	}
}

void Lu_oneapi(buffer<float, 2>& buf, queue& q) {

	//device my_device = q.get_device();
	//std::cout << "Device: " << my_device.get_info<info::device::name>() << std::endl;

	int n = buf.get_range()[0];
	for (int k = 0; k < n; k++) {

		q.submit([&](handler& h) {
			accessor m{ buf, h, read_write };
			h.parallel_for(range(n - k), [=](auto idx) {
				int j = k + idx;
				m[k][j] = m[k][j] / m[k][k];
				});
			});

		q.submit([&](handler& h) {
			accessor m{ buf, h, read_write };
			h.parallel_for(range(n - (k + 1), n - (k + 1)), [=](auto idx) {
				int i = k + 1 + idx.get_id(0);
				int j = k + 1 + idx.get_id(1);
				m[i][j] = m[i][j] - m[i][k] * m[k][j];
				});
			});

		q.submit([&](handler& h) {
			accessor m{ buf, h, read_write };
			h.parallel_for(range(n - (k + 1)), [=](auto idx) {
				int i = k + 1 + idx;
				m[i][k] = 0;
				});
			});
	}
	q.wait();
}


void test(int n, queue& q) {
	buffer<float, 2> buf1(range(n, n));
    buffer<float, 2> buf2(range(n, n));

	matrix_init(buf1);
    matrix_copy(buf1,buf1);
    auto start = std::chrono::high_resolution_clock::now();
    LU(buf1,q);
    auto end = std::chrono::high_resolution_clock::now();
    
    double time_ordinary = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout<<"ordinary time:"<<time_ordinary<<std::endl;

    start = std::chrono::high_resolution_clock::now();
    Lu_oneapi(buf2,q);
    end = std::chrono::high_resolution_clock::now();
    double time_oneapi = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout<<"oneapi time:"<<time_oneapi<<std::endl;

	return ;
}


int main() {

    //queue q(cpu_selector{}); //绑定到cpu
	queue q(gpu_selector{});   //绑定到gpu
	device my_device = q.get_device();
	std::cout << "Device: " << my_device.get_info<info::device::name>() << std::endl;

    int n=512;
    test(n,q);
}
