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
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>

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
    double learning_rate;
    int activation_function;
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

/**
 * a function, that saves the Neural Network in a file
 */
void saveNeuralNetwork(NeuralNetwork nn, const char* path);

/**
 * a function, that loads a Neural Network from a file
 */
NeuralNetwork loadNeuralNetwork(const char* path);

/* ================================================== */
/* The Matrix functions */
/* ================================================== */

Matrix* matrix_create(int rows, int cols, double* data) {
    Matrix* matrix = (Matrix*)malloc(sizeof(Matrix));
    matrix->rows = rows;
    matrix->cols = cols;
    matrix->data = data;

    return matrix;
}

/**
 * A function, that creates an empty matrix (all values are zero)
 */
Matrix* matrix_create_empty(int rows, int cols) {
    Matrix* matrix = (Matrix*)malloc(sizeof(Matrix));
    matrix->rows = rows;
    matrix->cols = cols;
    double* data = (double*) calloc(rows * cols, sizeof(double));
    matrix->data = data;

    return matrix;
}

/**
 * A function, that creates a copy of a matrix
 */
Matrix* matrix_copy(Matrix* mat) {
    Matrix* A = matrix_create_empty(mat->rows, mat->cols);
    memcpy(A->data, mat->data, sizeof(double) * mat->rows * mat->cols);
    return A;
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
 * A function, that transposes a matrix
 */
Matrix* matrix_transpose(Matrix* mat) {
    Matrix* A = matrix_create_empty(mat->cols, mat->rows);
    for(int i = 0; i < mat->rows; i++) {
        for(int j = 0; j < mat->cols; j++) {
            matrix_set(A, j, i, matrix_get(mat, i, j));
        }
    }
    return A;
}

/**
 * A function, that multiplies two matrices and returns the result in a new matrix 
 */
Matrix* matrix_multiply(Matrix* A, Matrix* B) {
    Matrix* C = matrix_create_empty(A->rows, B->cols);
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
void activationFunction(NeuralNetwork nn, Matrix* m) {
    if(nn.activation_function == NN_SIGMOID) {
        sigmoidFunction(m);
    } else {
        tanFunction(m);
    }
}

/**
 * The activation function, that is used to determine, which function to use
 */
void activationFunctiond(NeuralNetwork nn, Matrix* m) {
    if(nn.activation_function == NN_SIGMOID) {
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
    nn.learning_rate = NN_LEARNING_RATE;
    nn.activation_function = NN_ACTIVATION_FUNCTION;
    nn.input_nodes = input_nodes;
    nn.hidden_nodes = hidden_nodes;
    nn.output_nodes = output_nodes;
    nn.weights_ih = matrix_create_empty(hidden_nodes, input_nodes);
    nn.weights_ho = matrix_create_empty(output_nodes, hidden_nodes);
    nn.bias_h = matrix_create_empty(hidden_nodes, 1);
    nn.bias_o = matrix_create_empty(output_nodes, 1);

    // Fill in the matrices with random numbers between -1 and 1
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
    // Create input matrix
    Matrix* input = matrix_create_empty(nn.input_nodes, 1);
    for(int i = 0; i < nn.input_nodes; i++) {
        matrix_set(input, i, 0, in[i]);
    }

    // Transfer the input to the hidden nodes
    Matrix* hidden = matrix_multiply(nn.weights_ih, input);
    matrix_add(hidden, nn.bias_h);
    activationFunction(nn, hidden);

    // Transfer the data to the output nodes
    Matrix* output = matrix_multiply(nn.weights_ho, hidden);
    matrix_add(output, nn.bias_o);
    activationFunction(nn, output);

    // Return the values of the output nodes
    for(int i = 0; i < nn.output_nodes; i++) {
        out[i] = matrix_get(output, i, 0);
    }

    // Destroy matrices
    matrix_destroy(output);
    matrix_destroy(hidden);
    matrix_destroy(input);
}

void train(NeuralNetwork nn, double in[], double tar[]) {
    // *mario accent* mamma mia! time for spaghetti!
    
    // Predict the output (same as predict())
    Matrix* input = matrix_create_empty(nn.input_nodes, 1);
    for(int i = 0; i < nn.input_nodes; i++) {
        matrix_set(input, i, 0, in[i]);
    }

    Matrix* hidden = matrix_multiply(nn.weights_ih, input);
    matrix_add(hidden, nn.bias_h);
    activationFunction(nn, hidden);

    Matrix* output = matrix_multiply(nn.weights_ho, hidden);
    matrix_add(output, nn.bias_o);
    activationFunction(nn, output);

    // Create target matrix
    Matrix* target = matrix_create_empty(nn.output_nodes, 1);
    for(int i = 0; i < nn.output_nodes; i++) {
        matrix_set(target, i, 0, tar[i]);
    }

    // Calulate the error
    Matrix* output_error = matrix_copy(target);
    matrix_subtract(output_error, output);

    // Calculate gradient
    Matrix* gradient = matrix_copy(output);
    activationFunctiond(nn, gradient);
    matrix_multiply_elements(gradient, output_error);
    matrix_scale(gradient, nn.learning_rate);

    // Calculate deltas
    Matrix* hidden_T = matrix_transpose(hidden);
    Matrix* weights_ho_delta = matrix_multiply(gradient, hidden_T);

    // Adjust the weights by its deltas
    matrix_add(nn.weights_ho, weights_ho_delta);
    // Adjust the bias by its deltas (gradient)
    matrix_add(nn.bias_o, gradient);

    // Calculate the hidden layer errors
    Matrix* who_T = matrix_transpose(nn.weights_ho);
    Matrix* hidden_error = matrix_multiply(who_T, output_error);

    // Calculate hidden gradient
    Matrix* hidden_gradient = matrix_copy(hidden);
    activationFunctiond(nn, hidden_gradient);
    matrix_multiply_elements(hidden_gradient, hidden_error);
    matrix_scale(hidden_gradient, nn.learning_rate);

    // Calculate input->hidden deltas
    Matrix* input_T = matrix_transpose(input);
    Matrix* weights_ih_delta = matrix_multiply(hidden_gradient, input_T);

    // Adjust the weights by its deltas
    matrix_add(nn.weights_ih, weights_ih_delta);
    matrix_add(nn.bias_h, hidden_gradient);

    // Destroy matrices
    matrix_destroy(weights_ih_delta);
    matrix_destroy(input_T);
    matrix_destroy(hidden_gradient);
    matrix_destroy(hidden_error);
    matrix_destroy(who_T);
    matrix_destroy(weights_ho_delta);
    matrix_destroy(hidden_T);
    matrix_destroy(gradient);
    matrix_destroy(output_error);
    matrix_destroy(target);
    matrix_destroy(output);
    matrix_destroy(hidden);
    matrix_destroy(input);
}

void destroyNeuralNetwork(NeuralNetwork nn) {
    matrix_destroy(nn.weights_ih);
    matrix_destroy(nn.weights_ho);
    matrix_destroy(nn.bias_h);
    matrix_destroy(nn.bias_o);
}

void saveNeuralNetwork(NeuralNetwork nn, const char* path) {
    // Open / Create file
    FILE* file = fopen(path, "wb");

    // Write the id
    char id = 0b00101010;
    fwrite(&id, sizeof(char), 1, file);

    // Write NN_LEARNING_RATE
    double learningRate = nn.learning_rate;
    fwrite(&learningRate, sizeof(double), 1, file);

    // Write NN_ACTIVATION_FUNCTION
    int activation = nn.activation_function;
    fwrite(&activation, sizeof(int), 1, file);

    // Write input_nodes
    fwrite(&nn.input_nodes, sizeof(int), 1, file);

    // Write hidden_nodes
    fwrite(&nn.hidden_nodes, sizeof(int), 1, file);

    // Write output_nodes
    fwrite(&nn.output_nodes, sizeof(int), 1, file);

    // Write weights_ih
    fwrite(nn.weights_ih->data, sizeof(double), nn.input_nodes * nn.hidden_nodes, file);

    // Write weights_ho
    fwrite(nn.weights_ho->data, sizeof(double), nn.hidden_nodes * nn.output_nodes, file);

    // Write bias_h
    fwrite(nn.bias_h->data, sizeof(double), nn.hidden_nodes, file);

    // Write bias_h
    fwrite(nn.bias_o->data, sizeof(double), nn.output_nodes, file);

    fclose(file);
}

NeuralNetwork loadNeuralNetwork(const char* path) {
    // Open the file
    FILE* file = fopen(path, "rb");
    if(!file) { printf("File not found: %s\n", path); exit(-1); }

    // Check id
    char id;
    fread(&id, 1, 1, file);
    if(id != 0b00101010) {
        printf("Wrong format! Not the correct file?");
        fclose(file);
        exit(-1);
    }

    NeuralNetwork nn;

    // Get learning rate
    fread(&nn.learning_rate, sizeof(double), 1, file);

    // Get activation function
    fread(&nn.activation_function, sizeof(int), 1, file);

    // Get input nodes
    fread(&nn.input_nodes, sizeof(int), 1, file);

    // Get hidden nodes
    fread(&nn.hidden_nodes, sizeof(int), 1, file);

    // Get output nodes
    fread(&nn.output_nodes, sizeof(int), 1, file);

    // Get weights_ih
    double* ih_data = (double*)malloc(sizeof(double) * nn.input_nodes * nn.hidden_nodes);
    fread(ih_data, sizeof(double), nn.input_nodes * nn.hidden_nodes, file);
    nn.weights_ih = matrix_create(nn.hidden_nodes, nn.input_nodes, ih_data);

    // Get weights_ho
    double* ho_data = (double*)malloc(sizeof(double) * nn.hidden_nodes * nn.output_nodes);
    fread(ho_data, sizeof(double), nn.hidden_nodes * nn.output_nodes, file);
    nn.weights_ho = matrix_create(nn.output_nodes, nn.hidden_nodes, ho_data);

    // Get bias_h
    double* h_data = (double*)malloc(sizeof(double) * nn.hidden_nodes);
    fread(h_data, sizeof(double), nn.hidden_nodes, file);
    nn.bias_h = matrix_create(nn.hidden_nodes, 1, h_data);

    // Get bias_o
    double* o_data = (double*)malloc(sizeof(double) * nn.output_nodes);
    fread(o_data, sizeof(double), nn.output_nodes, file);
    nn.bias_o = matrix_create(nn.output_nodes, 1, o_data);

    fclose(file);

    return nn;
}
#endif /* NN_H */