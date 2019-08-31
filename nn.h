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

#ifndef NN_H
#define NN_H

#include <stdlib.h>
#include <math.h>
#include <time.h>

/* ================================================== */
/* General Settings */
/* ================================================== */

/**
 * The learning rate for backpropagation. Use #define NN_LEARNING_RATE [value] in your main file to change this value
 */
#ifndef NN_LEARNING_RATE
#define NN_LEARNING_RATE 0.1
#endif

/**
 * The avaiable activation functions
 */
enum ActivationFunctions {
    NN_SIGMOID,
    NN_TANGENT
};

/**
 * The used activation function. Use #define NN_ACTIVATION_FUNCTION [ActivationFunctions] in your main file to change the function
 */
#ifndef NN_ACTIVATION_FUNCTION
#define NN_ACTIVATION_FUNCTION NN_SIGMOID
#endif

/* ================================================== */
/* The main stuff, that should be used by the user */
/* ================================================== */

/**
 * The struct of the matrix (duh)
 */
typedef struct Matrix {
    int rows;
    int cols;
    double* data;
} Matrix;

/**
 * the struct, that inhabits the information of the Neural Network
 */
typedef struct NeuralNetwork {
    int input_nodes;
    int hidden_nodes;
    int output_nodes;
    Matrix* weights_ih;
    Matrix* weights_ho;
    Matrix* bias_h;
    Matrix* bias_o;
} NeuralNetwork;

/**
 * a function, that creates a Neural Network
 */
NeuralNetwork createNeuralNetwork(int input_nodes, int hidden_nodes, int output_nodes);

/**
 * a function, that calculates the output of the Neural Network. The length of the input array has to be the exact same as the amount of input nodes. The length of the output array has also to be exactlay the same as the amount of output nodes. The output values are always between 0 and 1
 */
void predict(NeuralNetwork nn, double in[], double* out);

/**
 * a function, that trains the Neural Network one time wit the given training input and the expected output. Input and output have to be the exact same length as the amount of their specific nodes
 */
void train(NeuralNetwork nn, double in[], double tar[]);

/**
 * a function, that destroys the Neural Network
 */
void destroyNeuralNetwork(NeuralNetwork nn);

/* ================================================== */
/* The Matrix functions */
/* ================================================== */

/**
 * A function, that creates a empty matrix (all values are zero)
 */
Matrix* matrix_create(int rows, int cols) {
    Matrix* matrix = (Matrix*)malloc(sizeof(Matrix));
    matrix->rows = rows;
    matrix->cols = cols;
    double* data = (double*) calloc(rows * cols, sizeof(double));
    matrix->data = data;

    return matrix;
}

/**
 * A function, that gets the value from a specific row and coloumn of a matrix
 */
double matrix_get(Matrix* mat, size_t row, size_t col) {
    return mat->data[row * mat->cols + col];
}

/**
 * A function, that sets the value from a specific row and coloumn of a matrix
 */
void matrix_set(Matrix* mat, size_t row, size_t col, double val) {
    mat->data[row * mat->cols + col] = val;
}

/**
 * A function, that adds matrix B to matrix A
 */
void matrix_add(Matrix* A, Matrix* B) {
    for(int i = 0; i < A->rows; i++) {
        for(int j = 0; j < A->cols; j++) {
            matrix_set(A, i, j, matrix_get(A, i, j) + matrix_get(B, i, j));
        }
    }
}

/**
 * A function, that subtracts matrix B from matrix A
 */
void matrix_subtract(Matrix* A, Matrix* B) {
    for(int i = 0; i < A->rows; i++) {
        for(int j = 0; j < A->cols; j++) {
            matrix_set(A, i, j, matrix_get(A, i, j) - matrix_get(B, i, j));
        }
    }
}

/**
 * A function, that multiplies each value of a matrix with a certain value 
 */
void matrix_scale(Matrix* A, double val) {
    for(int i = 0; i < A->rows; i++) {
        for(int j = 0; j < A->cols; j++) {
            matrix_set(A, i, j, matrix_get(A, i, j) * val);
        }
    }
}

/**
 * A function, that multiplies the elements of matrix B with the elements of matrix A
 */
void matrix_multiply_elements(Matrix* A, Matrix* B) {
    for(int i = 0; i < A->rows; i++) {
        for(int j = 0; j < A->cols; j++) {
            matrix_set(A, i, j, matrix_get(A, i, j) * matrix_get(B, i, j));
        }
    }
}

/**
 * A function, that transposes matrix B to matrix A
 */
void matrix_transpose(Matrix* A, Matrix* B) {
    for(int i = 0; i < A->rows; i++) {
        for(int j = 0; j < A->cols; j++) {
            matrix_set(A, j, i, matrix_get(B, i, j));
        }
    }
}

/**
 * A function, that multiplies two matrices and returns the result in a new matrix 
 */
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

/**
 * A function, that destroys a matrix
 */
void matrix_destroy(Matrix* mat) {
    free(mat->data);
    free(mat);
}

/* ================================================== */
/* Activation Functions */
/* ================================================== */

/**
 * The sigmoid function for x
 */
void sigmoidFunction(Matrix* m) {
    for(int i = 0; i < m->rows; i++) {
        for(int j = 0; j < m->cols; j++) {
            double val = matrix_get(m, i, j);
            matrix_set(m, i, j, 1 / (1 + exp(-val)));
        }
    }
}

/**
 * The sigmoid function for y
 */
void sigmoidFunctiond(Matrix* m) {
    for(int i = 0; i < m->rows; i++) {
        for(int j = 0; j < m->cols; j++) {
            double val = matrix_get(m, i, j);
            matrix_set(m, i, j, val * (1 - val));
        }
    }
}

/**
 * The tangent function for x
 */
void tanFunction(Matrix* m) {
    for(int i = 0; i < m->rows; i++) {
        for(int j = 0; j < m->cols; j++) {
            double val = matrix_get(m, i, j);
            matrix_set(m, i, j, tanh(val));
        }
    }
}

/**
 * The tangent function for y
 */
void tanFunctiond(Matrix* m) {
    for(int i = 0; i < m->rows; i++) {
        for(int j = 0; j < m->cols; j++) {
            double val = matrix_get(m, i, j);
            matrix_set(m, i, j, 1 - (val * val));
        }
    }
}

/**
 * The activation function, that is used to determine, which function to use
 */
void activationFunction(Matrix* m) {
    if(NN_ACTIVATION_FUNCTION == NN_SIGMOID) {
        sigmoidFunction(m);
    } else {
        tanFunction(m);
    }
}

/**
 * The activation function, that is used to determine, which function to use
 */
void activationFunctiond(Matrix* m) {
    if(NN_ACTIVATION_FUNCTION == NN_SIGMOID) {
        sigmoidFunctiond(m);
    } else {
        tanFunctiond(m);
    }
}

/* ================================================== */
/* The main functions */
/* ================================================== */

NeuralNetwork createNeuralNetwork(int input_nodes, int hidden_nodes, int output_nodes) {
    NeuralNetwork nn;
    nn.input_nodes = input_nodes;
    nn.hidden_nodes = hidden_nodes;
    nn.output_nodes = output_nodes;
    nn.weights_ih = matrix_create(hidden_nodes, input_nodes);
    nn.weights_ho = matrix_create(output_nodes, hidden_nodes);
    nn.bias_h = matrix_create(hidden_nodes, 1);
    nn.bias_o = matrix_create(output_nodes, 1);

    srand(time(NULL));

    for(int i = 0; i < hidden_nodes; i++) {
        for(int j = 0; j < input_nodes; j++) {
            matrix_set(nn.weights_ih, i, j, (double)rand()/RAND_MAX * 2 - 1);
        }
    }
    for(int i = 0; i < output_nodes; i++) {
        for(int j = 0; j < hidden_nodes; j++) {
            matrix_set(nn.weights_ho, i, j, (double)rand()/RAND_MAX * 2 - 1);
        }
    }
    for(int i = 0; i < hidden_nodes; i++) {
        matrix_set(nn.bias_h, i, 0, (double)rand()/RAND_MAX * 2 - 1);
    }
    for(int i = 0; i < output_nodes; i++) {
        matrix_set(nn.bias_o, i, 0, (double)rand()/RAND_MAX * 2 - 1);
    }

    return nn;
}

void predict(NeuralNetwork nn, double in[], double* out) {
    Matrix* input = matrix_create(nn.input_nodes, 1);
    for(int i = 0; i < nn.input_nodes; i++) {
        matrix_set(input, i, 0, in[i]);
    }

    Matrix* hidden = matrix_multiply(nn.weights_ih, input);
    matrix_add(hidden, nn.bias_h);
    activationFunction(hidden);

    Matrix* output = matrix_multiply(nn.weights_ho, hidden);
    matrix_add(output, nn.bias_o);
    activationFunction(output);

    matrix_destroy(hidden);
    matrix_destroy(input);

    for(int i = 0; i < nn.output_nodes; i++) {
        out[i] = matrix_get(output, i, 0);
    }
}

void train(NeuralNetwork nn, double in[], double tar[]) {
    // TODO: Needs Documentation (for myself)
    // Predict the output
    Matrix* input = matrix_create(nn.input_nodes, 1);
    for(int i = 0; i < nn.input_nodes; i++) {
        matrix_set(input, i, 0, in[i]);
    }

    Matrix* hidden = matrix_multiply(nn.weights_ih, input);
    matrix_add(hidden, nn.bias_h);
    activationFunction(hidden);

    Matrix* output = matrix_multiply(nn.weights_ho, hidden);
    matrix_add(output, nn.bias_o);
    activationFunction(output);

    // Format the target values
    Matrix* target = matrix_create(nn.output_nodes, 1);
    for(int i = 0; i < nn.output_nodes; i++) {
        matrix_set(target, i, 0, tar[i]);
    }

    // Get the Error
    matrix_subtract(target, output);

    // Let weights_ho and bias_ho learn from this error
    activationFunctiond(output);
    matrix_multiply_elements(output, target);
    matrix_scale(output, NN_LEARNING_RATE);

    Matrix* hiddenT = matrix_create(hidden->cols, hidden->rows);
    matrix_transpose(hiddenT, hidden);
    Matrix* weight_ho_delta = matrix_multiply(output, hiddenT);
    matrix_destroy(hiddenT);

    matrix_add(nn.weights_ho, weight_ho_delta);
    matrix_destroy(weight_ho_delta);
    matrix_add(nn.bias_o, output);
    matrix_destroy(output);

    // Let weights_ih and bias_ih learn from the error
    Matrix* whoT = matrix_create(nn.weights_ho->cols, nn.weights_ho->rows);
    matrix_transpose(whoT, nn.weights_ho);
    Matrix* hidden_error = matrix_multiply(whoT, target);
    matrix_destroy(whoT);

    activationFunctiond(hidden);
    matrix_multiply_elements(hidden, hidden_error);
    matrix_destroy(hidden_error);
    matrix_scale(hidden, NN_LEARNING_RATE);

    Matrix* inputT = matrix_create(input->cols, input->rows);
    matrix_transpose(inputT, input);
    matrix_destroy(input);
    Matrix* weight_ih_delta = matrix_multiply(hidden, inputT);
    matrix_destroy(inputT);
    
    matrix_add(nn.weights_ih, weight_ih_delta);
    matrix_destroy(weight_ih_delta);
    matrix_add(nn.bias_h, hidden);

    matrix_destroy(hidden);
}

void destroyNeuralNetwork(NeuralNetwork nn) {
    matrix_destroy(nn.weights_ih);
    matrix_destroy(nn.weights_ho);
    matrix_destroy(nn.bias_h);
    matrix_destroy(nn.bias_o);
}
#endif /* NN_H */