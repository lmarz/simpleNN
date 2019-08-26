/* MIT License

Copyright (c) 2019 lmarz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. */

#include <stdlib.h>

#ifndef MATRIX_H
#define MATRIX_H

typedef struct Matrix {
    int rows;
    int cols;
    double* data;
} Matrix;

Matrix* matrix_create(int rows, int cols) {
    Matrix* matrix = (Matrix*)malloc(sizeof(Matrix));
    matrix->rows = rows;
    matrix->cols = cols;
    double* data = (double*) calloc(rows * cols, sizeof(double));
    matrix->data = data;

    return matrix;
}

double matrix_get(Matrix* mat, size_t row, size_t col) {
    return mat->data[row * mat->cols + col];
}

void matrix_set(Matrix* mat, size_t row, size_t col, double val) {
    mat->data[row * mat->cols + col] = val;
}

void matrix_add(Matrix* A, Matrix* B) {
    for(int i = 0; i < A->rows; i++) {
        for(int j = 0; j < A->cols; j++) {
            matrix_set(A, i, j, matrix_get(A, i, j) + matrix_get(B, i, j));
        }
    }
}

void matrix_subtract(Matrix* A, Matrix* B) {
    for(int i = 0; i < A->rows; i++) {
        for(int j = 0; j < A->cols; j++) {
            matrix_set(A, i, j, matrix_get(A, i, j) - matrix_get(B, i, j));
        }
    }
}

void matrix_scale(Matrix* A, double val) {
    for(int i = 0; i < A->rows; i++) {
        for(int j = 0; j < A->cols; j++) {
            matrix_set(A, i, j, matrix_get(A, i, j) * val);
        }
    }
}

void matrix_multiply_elements(Matrix* A, Matrix* B) {
    for(int i = 0; i < A->rows; i++) {
        for(int j = 0; j < A->cols; j++) {
            matrix_set(A, i, j, matrix_get(A, i, j) * matrix_get(B, i, j));
        }
    }
}

void matrix_transpose(Matrix* A, Matrix* B) {
    for(int i = 0; i < A->rows; i++) {
        for(int j = 0; j < A->cols; j++) {
            matrix_set(A, j, i, matrix_get(B, i, j));
        }
    }
}

Matrix* matrix_multiply(Matrix* A, Matrix* B) {
    Matrix* C = matrix_create(A->rows, B->cols);
    for(int i = 0; i < C->rows; i++) {
        for(int j = 0; j < C->cols; j++) {
            for(int k = 0; k < A->cols; k++) {
                matrix_set(C, i, j, matrix_get(C, i, j) + matrix_get(A, i, k) * matrix_get(B, k, j));
            }
        }
    }

    return C;
}

void matrix_destroy(Matrix* mat) {
    free(mat->data);
    free(mat);
}
#endif /* MATRIX_H */