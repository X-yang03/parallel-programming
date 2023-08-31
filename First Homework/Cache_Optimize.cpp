#include <iostream>

using namespace std;
#define MAXN 5000
int b[MAXN][MAXN];
    int a[MAXN];
    int sum[MAXN];

int main()
{

    for(int i=0;i<MAXN;i++)
    {
        for(int j=0;j<MAXN;j++)
        {
            b[i][j]=i+j;
        }
    }
    for(int i=0;i<MAXN;i++)
    {
        a[i]=i;
    }

   /* 
 {
        for(int i=0;i<MAXN;i++)
        {
            sum[i]=0;
            for(int j=0;j<MAXN;j++)  //平凡算法
            {
                sum[i]+=b[j][i]*a[j];
            }
        }
    }

*/


    {
        for(int i=0;i<MAXN;i++)
        {
            sum[i]=0;

        }
        for(int j=0;j<MAXN;j++)
        {
            for(int i=0;i<MAXN;i++)         //Cache优化
            {
                sum[i]+=b[j][i]*a[j];
            }
        }
    }


    return 0;
}
