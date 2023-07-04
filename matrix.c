#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

double** alloc_mat(int col, int row) {
    double * A = malloc(row * col * sizeof(double));
    double ** p = malloc(row * sizeof(double*));
    for (int i = 0; i < row; i++) {
        p[i] = A + (i * col * sizeof(double));
    }
    return p;
}

void free_mat(double **A) {
    free(*A);
    free(A);
}

void alloc_spmat(int nnz, int** col, int** row, double **A){
    *col = malloc(nnz * sizeof (int));
    *row = malloc(nnz * sizeof (int));
    *A = malloc(nnz * sizeof (double));
}

void outer_product(int n, int m, double *u, double *v, double **A){
    for (int i = 0; i<n; i++){
        for (int j = 0; j<m; j++){
            A[i][j] = u[i] * v[j];
        }
    }
}

double inner_product(int n, double *u, double *v) {
    double b = 0;
    for (int j = 0; j < n; ++j) {
        b += u[j] * v[j];
    }
    return b;
}

void matrix_vector(int n1, int n2, double **A, double *x, double *b) {
    double sum;
    for (int i = 0; i<n1; i++) {
        sum = 0;
        for (int j = 0; j<n2; j++) {
            sum += A[i][j] * x[j];
        }
        b[i] = sum;
    }
}

void sparsemv(int n1, int n2, int nnz, int *r, int *c, double *a, double *x, double *b){
    for (int i = 0; i<nnz; i++){
        b[r[i]] += a[c[i]] * x[r[i]];
    }
}

void test_outer_product() {
    double * u = malloc(3 * sizeof(double));
    double * v = malloc(3 * sizeof(double));
    for (int i = 0; i < 3; ++i) {
        u[i] = 2;
    }
    for (int i = 0; i < 3; ++i) {
        v[i] = 3;
    }
    double ** A = alloc_mat(3, 3);
    outer_product(3, 3, u, v, A);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            printf("%f ", A[i][j]);
        }
        printf("\n");
    }
}

void test_inner_product() {
    double * u = malloc(3 * sizeof(double));
    double * v = malloc(3 * sizeof(double));
    for (int i = 0; i < 3; ++i) {
        u[i] = 2;
    }
    for (int i = 0; i < 3; ++i) {
        v[i] = 3;
    }
    printf("%f", inner_product(3, u, v));
}

void test_matrix_vector() {
    double ** A = alloc_mat(3, 3);
    double * x = malloc(3 * sizeof(double));
    double * b = malloc(3 * sizeof(double));
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            A[i][j] = 3;
        }
    }
    for (int i = 0; i < 3; ++i) {
        x[i] = 2;
    }
    matrix_vector(3, 3, A, x, b);
    for (int i = 0; i < 3; ++i) {
        printf("%f\n", b[i]);
    }
}

void test_sparsemv() {
    double ** A = malloc(sizeof(double*));
    int ** col = malloc(sizeof(int*));
    int ** row = malloc(sizeof(int*));
    alloc_spmat(6, col, row, A);
    A[0][0] = 3;
    col[0][0] = 1;
    row[0][0] = 0;
    A[0][1] = 3;
    col[0][1] = 0;
    row[0][1] = 1;
    A[0][2] = 3;
    col[0][2] = 2;
    row[0][2] = 1;
    A[0][3] = 3;
    col[0][3] = 0;
    row[0][3] = 2;
    A[0][4] = 3;
    col[0][4] = 1;
    row[0][4] = 2;
    A[0][5] = 3;
    col[0][5] = 2;
    row[0][5] = 2;
    double * x = malloc(3 * sizeof(double));
    x[0] = 1;
    x[1] = 2;
    x[2] = 3;
    double * b = malloc(3 * sizeof(double));
    for (int i = 0; i < 3; ++i) {
        b[i] = 0;
    }
    sparsemv(0, 0, 6, row[0], col[0], A[0], x, b);
    for (int i = 0; i < 3; ++i) {
        printf("%f\n", b[i]);
    }
}

double parallel_inner_product(int n, double *u, double *v) {
    double b = 0;
#pragma omp parallel for reduction(+:b)
    for (int j = 0; j < n; ++j) {
        b += u[j] * v[j];
    }
    return b;
}

void parallel_outer_product(int n, int m, double *u, double *v, double **A) {
#pragma parallel for
    for (int i = 0; i<n; i++){
#pragma parallel for
        for (int j = 0; j<m; j++){
            A[i][j] = u[i] * v[j];
        }
    }
}

void parallel_matrix_vector(int n1, int n2, double **A, double *x, double *b) {
#pragma parallel for
    for (int i = 0; i < n1; ++i) {
#pragma parallel for
        for (int j = 0; j < n2; ++j) {

        }
    }
}

int main() {
    for (int i = 1; i < 50; ++i) {
        double * v = malloc(i * sizeof(double));
        double * u = malloc(i + sizeof(double));
        double ** A = alloc_mat(i, i);
        free(u);
        free(v);
        free_mat(A);
    }
    return 0;
}
