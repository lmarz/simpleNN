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

#include <gsl/gsl_matrix.h>
#include <math.h>
#include <time.h>

#ifndef NN_H
#define NN_H

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
/* Helper Functions */
/* ================================================== */

/**
 * A function, that multiplies two matrices and outputs a new one
 */
gsl_matrix* multiplyMatrices(gsl_matrix* a, gsl_matrix* b) {
    if(a->size2 != b->size1) return NULL;
    gsl_matrix* c = gsl_matrix_alloc(a->size1, b->size2);
    gsl_matrix_set_zero(c);
    for(int i = 0; i < c->size1; i++) {
        for(int j = 0; j < c->size2; j++) {
            for(int k = 0; k < a->size2; k++) {
                gsl_matrix_set(c, i, j, gsl_matrix_get(c, i, j) + gsl_matrix_get(a, i, k) * gsl_matrix_get(b, k, j));
            }
        }
    }

    return c;
}

/**
 * The sigmoid function for x
 */
void sigmoidFunction(gsl_matrix* m) {
    for(int i = 0; i < m->size1; i++) {
        for(int j = 0; j < m->size2; j++) {
            double val = gsl_matrix_get(m, i, j);
            gsl_matrix_set(m, i, j, 1 / (1 + exp(-val)));
        }
    }
}

/**
 * The sigmoid function for y
 */
void sigmoidFunctiond(gsl_matrix* m) {
    for(int i = 0; i < m->size1; i++) {
        for(int j = 0; j < m->size2; j++) {
            double val = gsl_matrix_get(m, i, j);
            gsl_matrix_set(m, i, j, val * (1 - val));
        }
    }
}

/**
 * The tangent function for x
 */
void tanFunction(gsl_matrix* m) {
    for(int i = 0; i < m->size1; i++) {
        for(int j = 0; j < m->size2; j++) {
            double val = gsl_matrix_get(m, i, j);
            gsl_matrix_set(m, i, j, tanh(val));
        }
    }
}

/**
 * The tangent function for y
 */
void tanFunctiond(gsl_matrix* m) {
    for(int i = 0; i < m->size1; i++) {
        for(int j = 0; j < m->size2; j++) {
            double val = gsl_matrix_get(m, i, j);
            gsl_matrix_set(m, i, j, 1 - (val * val));
        }
    }
}

/**
 * The activation function, that is used to determine, which function to use
 */
void activationFunction(gsl_matrix* m) {
    if(NN_ACTIVATION_FUNCTION == NN_SIGMOID) {
        sigmoidFunction(m);
    } else {
        tanFunction(m);
    }
}

/**
 * The activation function, that is used to determine, which function to use
 */
void activationFunctiond(gsl_matrix* m) {
    if(NN_ACTIVATION_FUNCTION == NN_SIGMOID) {
        sigmoidFunctiond(m);
    } else {
        tanFunctiond(m);
    }
}

/* ================================================== */
/* The main stuff, that should be used by the user */
/* ================================================== */

/**
 * the struct, that inhabits the information of the Neural Network
 */
typedef struct NeuralNetwork {
    int input_nodes;
    int hidden_nodes;
    int output_nodes;
    gsl_matrix* weights_ih;
    gsl_matrix* weights_ho;
    gsl_matrix* bias_h;
    gsl_matrix* bias_o;
} NeuralNetwork;

/**
 * a function, that creates a Neural Network
 */
NeuralNetwork createNeuralNetwork(int input_nodes, int hidden_nodes, int output_nodes) {
    NeuralNetwork nn;
    nn.input_nodes = input_nodes;
    nn.hidden_nodes = hidden_nodes;
    nn.output_nodes = output_nodes;
    nn.weights_ih = gsl_matrix_alloc(hidden_nodes, input_nodes);
    nn.weights_ho = gsl_matrix_alloc(output_nodes, hidden_nodes);
    nn.bias_h = gsl_matrix_alloc(hidden_nodes, 1);
    nn.bias_o = gsl_matrix_alloc(output_nodes, 1);

    srand(time(NULL));

    for(int i = 0; i < hidden_nodes; i++) {
        for(int j = 0; j < input_nodes; j++) {
            gsl_matrix_set(nn.weights_ih, i, j, (double)rand()/RAND_MAX * 2 - 1);
        }
    }
    for(int i = 0; i < output_nodes; i++) {
        for(int j = 0; j < hidden_nodes; j++) {
            gsl_matrix_set(nn.weights_ho, i, j, (double)rand()/RAND_MAX * 2 - 1);
        }
    }
    for(int i = 0; i < hidden_nodes; i++) {
        gsl_matrix_set(nn.bias_h, i, 0, (double)rand()/RAND_MAX * 2 - 1);
    }
    for(int i = 0; i < output_nodes; i++) {
        gsl_matrix_set(nn.bias_o, i, 0, (double)rand()/RAND_MAX * 2 - 1);
    }

    return nn;
}

/**
 * a function, that calculates the output of the Neural Network. The length of the input array has to be the exact same as the amount of input nodes. The length of the output array has also to be exactlay the same as the amount of output nodes. The output values are always between 0 and 1
 */
void predict(NeuralNetwork nn, double in[], double* out) {
    gsl_matrix* input = gsl_matrix_alloc(nn.input_nodes, 1);
    for(int i = 0; i < nn.input_nodes; i++) {
        gsl_matrix_set(input, i, 0, in[i]);
    }

    gsl_matrix* hidden = multiplyMatrices(nn.weights_ih, input);
    gsl_matrix_add(hidden, nn.bias_h);
    activationFunction(hidden);

    gsl_matrix* output = multiplyMatrices(nn.weights_ho, hidden);
    gsl_matrix_add(output, nn.bias_o);
    activationFunction(output);

    gsl_matrix_free(hidden);
    gsl_matrix_free(input);

    for(int i = 0; i < nn.output_nodes; i++) {
        out[i] = gsl_matrix_get(output, i, 0);
    }
}

/**
 * a function, that trains the Neural Network one time wit the given training input and the expected output. Input and output have to be the exact same length as the amount of their specific nodes
 */
void train(NeuralNetwork nn, double in[], double tar[]) {
    // TODO: Needs Documentation (for myself)
    // Predict the output
    gsl_matrix* input = gsl_matrix_alloc(nn.input_nodes, 1);
    for(int i = 0; i < nn.input_nodes; i++) {
        gsl_matrix_set(input, i, 0, in[i]);
    }

    gsl_matrix* hidden = multiplyMatrices(nn.weights_ih, input);
    gsl_matrix_add(hidden, nn.bias_h);
    activationFunction(hidden);

    gsl_matrix* output = multiplyMatrices(nn.weights_ho, hidden);
    gsl_matrix_add(output, nn.bias_o);
    activationFunction(output);

    // Format the target values
    gsl_matrix* target = gsl_matrix_alloc(nn.output_nodes, 1);
    for(int i = 0; i < nn.output_nodes; i++) {
        gsl_matrix_set(target, i, 0, tar[i]);
    }

    // Get the Error
    gsl_matrix_sub(target, output);

    // Let weights_ho and bias_ho learn from this error
    activationFunctiond(output);
    gsl_matrix_mul_elements(output, target);
    gsl_matrix_scale(output, NN_LEARNING_RATE);

    gsl_matrix* hiddenT = gsl_matrix_alloc(hidden->size2, hidden->size1);
    gsl_matrix_transpose_memcpy(hiddenT, hidden);
    gsl_matrix* weight_ho_delta = multiplyMatrices(output, hiddenT);
    gsl_matrix_free(hiddenT);

    gsl_matrix_add(nn.weights_ho, weight_ho_delta);
    gsl_matrix_free(weight_ho_delta);
    gsl_matrix_add(nn.bias_o, output);
    gsl_matrix_free(output);

    // Let weights_ih and bias_ih learn from the error
    gsl_matrix* whoT = gsl_matrix_alloc(nn.weights_ho->size2, nn.weights_ho->size1);
    gsl_matrix_transpose_memcpy(whoT, nn.weights_ho);
    gsl_matrix* hidden_error = multiplyMatrices(whoT, target);
    gsl_matrix_free(whoT);

    activationFunctiond(hidden);
    gsl_matrix_mul_elements(hidden, hidden_error);
    gsl_matrix_free(hidden_error);
    gsl_matrix_scale(hidden, NN_LEARNING_RATE);

    gsl_matrix* inputT = gsl_matrix_alloc(input->size2, input->size1);
    gsl_matrix_transpose_memcpy(inputT, input);
    gsl_matrix_free(input);
    gsl_matrix* weight_ih_delta = multiplyMatrices(hidden, inputT);
    gsl_matrix_free(inputT);
    
    gsl_matrix_add(nn.weights_ih, weight_ih_delta);
    gsl_matrix_free(weight_ih_delta);
    gsl_matrix_add(nn.bias_h, hidden);

    gsl_matrix_free(hidden);
}

/**
 * a function, that destroys the Neural Network
 */
void destroyNeuralNetwork(NeuralNetwork nn) {
    gsl_matrix_free(nn.weights_ih);
    gsl_matrix_free(nn.weights_ho);
    gsl_matrix_free(nn.bias_h);
    gsl_matrix_free(nn.bias_o);
}
#endif /* NN_H */