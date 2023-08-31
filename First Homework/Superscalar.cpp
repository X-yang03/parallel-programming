#include<iostream>
using namespace std;
#define MAXN 100000000
int a[MAXN];
int sum,sum1,sum2=0;

int main()
{
    int t=50;
    for(int i=0;i<MAXN;i++)
    {
        a[i]=i;
    }  //test git
    while(t--){
     for(int i=0;i<MAXN;i++)   //平凡算法
    {
        sum+=a[i];
    }
    }



    /*
    while(t--){
     for(int i=0;i<MAXN;i+=2)
    {
        sum1+=a[i];          //优化算法
        sum2+=a[i+1];
    }
    }
    sum=sum1+sum2;
    */


}
