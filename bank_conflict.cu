#include<iostream>
#include<cuda.h>
#include<time.h>
using namespace std;

__global__ void kernel( unsigned long long* d_Time, int* d_a, int* foo) {
    __shared__ float MySharedMemory[8192];
    unsigned long long Time_start;
    unsigned long long Time_stop;
    Time_start = clock();
    if(threadIdx.x%2 == 0){
        d_a[0]++;
    } else{
        d_a[*foo]++;
    }
    Time_stop = clock();
    *d_Time = (Time_stop-Time_start);
}
int main( void ) {
    int *d_foo;
    int* foo = (int*) malloc(sizeof(int));
    cudaMalloc((void**) &d_foo,sizeof(int));
    int reps=10;
    unsigned long long Time;
    unsigned long long* d_Time;
    cudaMalloc((void**)&d_Time,sizeof(unsigned long long));
    int* a;
    int* d_a;
    cudaMalloc((void**)&d_a, sizeof(int)*33);

    cudaMemcpy(d_Time,&Time, sizeof(unsigned long long), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a,a, sizeof(int)*33, cudaMemcpyHostToDevice);
    *foo = 32;
    cudaMemcpy(d_foo,foo,sizeof(int),cudaMemcpyHostToDevice);
    kernel<<<1,32>>>(d_Time, d_a, d_foo);
    cudaMemcpy(&Time,d_Time, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cout<<"Elapsed Time with conflict:    "<<(Time)<<endl;
    *foo = 31;
    cudaMemcpy(d_foo,foo,sizeof(int),cudaMemcpyHostToDevice);
    kernel<<<1,32>>>(d_Time, d_a, d_foo);
    cudaMemcpy(&Time,d_Time, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cout<<"Elapsed Time without conflict: "<<(Time)<<endl;
    cudaFree(d_Time);
    return 0;
}
