#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

int main() {
    int N = 100000000;
    int I = 0;
    double x,y,l, pi;
    srand( time(NULL) );
    double begin = omp_get_wtime();
    for(int i = 0; i < N; i++) {
        x = (double)rand() / RAND_MAX;
        y = (double)rand() / RAND_MAX;
        l = sqrt(x * x + y * y);
        if( l <= 1 ) I++;
    }
    pi = (double) I / N * 4;
    double end = omp_get_wtime();
    printf("Approximated non parallel PI = %g\n", pi);
    printf("Non parallel time %g\n", end-begin);

    N = 100000000;
    I = 0;
    srand( time(NULL) );
    begin = omp_get_wtime();
#pragma omp parallel for private(x,y,l) num_threads(100)
    for(int i = 0; i < N; i++) {
        x = (double)rand() / RAND_MAX;
        y = (double)rand() / RAND_MAX;
        l = sqrt(x * x + y * y);
        if( l <= 1 ) {
#pragma omp atomic update
            I++;
        }
    }
    pi = (double) I / N * 4;
    end = omp_get_wtime();
    printf("Approximated parallel atomic PI = %g\n", pi);
    printf("Parallel atomic time %g\n", end-begin);

    N = 100000000;
    I = 0;
    srand( time(NULL) );
    begin = omp_get_wtime();
#pragma omp parallel for reduction(+:I) private(x,y,l) num_threads(100)
    for(int i = 0; i < N; i++) {
        x = (double)rand() / RAND_MAX;
        y = (double)rand() / RAND_MAX;
        l = sqrt(x * x + y * y);
        if( l <= 1 ) {
            I++;
        }
    }
    pi = (double) I / N * 4;
    end = omp_get_wtime();
    printf("Approximated parallel reduction PI = %g\n", pi);
    printf("Parallel reduction time %g\n", end-begin);

    N = 100000000;
    I = 0;
    srand( time(NULL) );
    begin = omp_get_wtime();
#pragma omp parallel for reduction(+:I) private(x,y,l) num_threads(100)
    for(int i = 0; i < N; i++) {
        x = (double)rand() / RAND_MAX;
        y = (double)rand() / RAND_MAX;
        l = sqrt(x * x + y * y);
        if( l <= 1 ) {
#pragma omp atomic update
            I++;
        }
    }
    pi = (double) I / N * 4;
    end = omp_get_wtime();
    printf("Approximated parallel both PI = %g\n", pi);
    printf("Parallel both time %g\n", end-begin);

    N = 100000000;
    I = 0;
    srand( time(NULL) );
    begin = omp_get_wtime();
#pragma omp parallel for private(x,y,l) num_threads(100)
    for(int i = 0; i < N; i++) {
        x = (double)rand() / RAND_MAX;
        y = (double)rand() / RAND_MAX;
        l = sqrt(x * x + y * y);
        if( l <= 1 ) {
#pragma omp critical
            I++;
        }
    }
    pi = (double) I / N * 4;
    end = omp_get_wtime();
    printf("Approximated parallel critical PI = %g\n", pi);
    printf("Parallel critical time %g\n", end-begin);

    return 0;
}
