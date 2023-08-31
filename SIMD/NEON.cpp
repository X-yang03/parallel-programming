
#include<iostream>
#include<time.h>
#include<arm_neon.h>
using namespace std;
#define N 4000
float m[N][N];
float **a;
void m_reset(){
    for(int i = 0,j=0 ; i < N ; i++){   //生成数组
        m[i][i] = 1.0;
        for(int j = i+1; j < N; j++){
            m[i][j] = rand()%1000;
        }

    }
    for(int k = 0; k < N; k++){
        for(int i = k+1; i < N; i++){
            for(int j = 0; j < N; j++){
                m[i][j] += m[k][j];
                m[i][j] = (int)m[i][j]%1000;
            }
        }
    }
}

void copyMatrix(float** a, float m[N][N]){
	for(int i=0;i<N;i++){
		for(int j=0;j<N;j++)
			a[i][j]=m[i][j];
	}
}

void deleteMatrix(float** a){
	for(int i=0;i<N;i++)
		delete[] a[i];
	delete a;
}

void print(){
   for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
        cout<<m[i][j]<<" ";
    }
    cout<<endl;
   }
} 
void LU(float** m){      //平凡算法
    for(int k = 0; k<N; k++){
        for(int j = k+1; j<N; j++ ){
            m[k][j] = m[k][j]/m[k][k];
        }
        m[k][k] = 1.0;
        for(int i = k+1; i<N; i++){
            for(int j = k+1; j<N; j++){
                m[i][j] = m[i][j]-m[i][k]*m[k][j];
            }
            m[i][k] = 0;
        }
    }
}

void neon_optimized(float** m){            //neon优化算法
    for(int k = 0; k < N; k++){
        float32x4_t vt = vdupq_n_f32(m[k][k]);
        int j = 0;
        for(j = k+1; j+4 <= N; j+=4){
                float32x4_t va = vld1q_f32(&m[k][j]);
                va = vdivq_f32(va, vt);
                vst1q_f32(&m[k][j], va);
        }
        for( ;j < N; j++){
                m[k][j] = m[k][j]/m[k][k];
        }
        m[k][k] = 1.0;
        for(int i = k+1; i < N; i++){
                float32x4_t vaik = vdupq_n_f32(m[i][k]);
                for(j = k+1; j+4 <= N; j+=4){
                        float32x4_t vakj = vld1q_f32(&m[k][j]);
                        float32x4_t vaij = vld1q_f32(&m[i][j]);
                        float32x4_t vx = vmulq_f32(vakj, vaik);
                        vaij = vsubq_f32(vaij , vx);
                        vst1q_f32(&m[i][j] , vaij);
        }
        for( ; j < N; j++){
                m[i][j] = m[i][j]-m[i][k]*m[k][j];
        }
        m[i][k] = 0;

    }
}
}


int main(){
    m_reset();
    a = new float*[N];
    for(int i=0;i<N;i++)
         a[i] = new float[N];
    copyMatrix(a,m);
    struct timespec sts,ets;
   // print();
    timespec_get(&sts, TIME_UTC);
     LU(a);
    timespec_get(&ets, TIME_UTC);
    time_t dsec = ets.tv_sec-sts.tv_sec;
    long dnsec = ets.tv_nsec - sts.tv_nsec;
    if(dnsec<0){
        dsec--;
        dnsec+=1000000000ll;
   }
   

   // print();
    printf("平凡算法耗时： %ld.%09lds\n",dsec,dnsec);
    copyMatrix(a,m);
    timespec_get(&sts,TIME_UTC);
    neon_optimized(a);
    timespec_get(&ets,TIME_UTC);
    dsec = ets.tv_sec - sts.tv_sec;
    dnsec = ets.tv_nsec - sts.tv_nsec;
    if(dnsec < 0){
        dsec--;
        dnsec += 1000000000ll;
    }
    printf("NEON优化后耗时： %ld.%09lds\n",dsec,dnsec);
   
    
}

                                                               

