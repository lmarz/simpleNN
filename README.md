# simpleNN
Simple Neural Network header written in C

Inspired by TheCodingTrain - [Toy-Neural-Network-JS](https://github.com/CodingTrain/Toy-Neural-Network-JS)

It has one hidden layer and uses Backpropagation as learning method.

## Getting Started
### Prerequisites
Just copy nn.h into your working directory 

### Compiling
For gcc, you have to include these libraries in the linking process:
```
-lm
```

### Documentation
#### Macros
* `NN_LEARNING_RATE` - the learning rate for backpropagation. Default is `0.1`. Use `#define NN_LEARNING_RATE [double]` in your main file **before #include "nn.h"** to change this value
* `NN_ACTIVATION_FUNCTION` - the used activation function Default is `NN_SIGMOID`. Use `#define NN_ACTIVATION_FUNCTION [ActivationFunctions]` in your main file **before #include "nn.h"** to change the function

#### Structs and Enums
* `enum ActivationFunctions` - the avaiable activation functions
    * `NN_SIGMOID` - the sigmoid function
    * `NN_TANGENT` - the tangent function
* `struct NeuralNetwork` - the struct, that inhabits the information of the Neural Network
    * `double learning_rate`
    * `int input_nodes` - the amount of input nodes
    * `int hidden_nodes` - the amount of hidden nodes
    * `int output_nodes` - the amount of output nodes
    * `Matrix* weights_ih` - the weights of the connections between the input nodes and the hidden nodes
    * `Matrix* weights_ho` - the weights of the connections between the hidden nodes and the output nodes
    * `Matrix* bias_ih` - the bias of the connections between the input nodes and the hidden nodes
    * `Matrix* bias_ho` - the bias of the connections between the hidden nodes and the output nodes

#### Functions
* `NeuralNetwork createNeuralNetwork(int input_nodes, int hidden_nodes, int output_nodes)` - a function, that creates a Neural Network
* `void predict(NeuralNetwork nn, double[] input, double[] output)` - a function, that calculates the output of the Neural Network. The length of the input array has to be the exact same as the amount of input nodes. The length of the output array has also to be exactlay the same as the amount of output nodes. The output values are always between 0 and 1
* `void train(NeuralNetwork nn, double[] training_input, double[] training_output)` - a function, that trains the Neural Network one time with the given training input and the expected output. Input and output have to be the exact same length as the amount of their specific nodes
* `void destroyNeuralNetwork(NeuralNetwork nn)` - a function, that destroys the Neural Network
* `void saveNeuralNetwork(NeuralNetwork nn, const char* path)` - a function, that saves the Neural Network in a file. For more details, see [fileformat](fileformat/fileformat.md)
* `NeuralNetwork loadNeuralNetwork(const char* path)` - a function, that loads a Neural Network from a file. For more details, see [fileformat](fileformat/fileformat.md)

## Example
Here is a simple example on how to use the Neural Network:

``` c
#include "nn.h"

int main() {
    // Specify the amount of Nodes for the Neural Network
    int input_nodes = 2;
    int hidden_nodes = 1;
    int output_nodes = 1;

    // Create the Neural Network
    NeuralNetwork nn = createNeuralNetwork(input_nodes, hidden_nodes, output_nodes);

    // Specify the training data
    double training_input_1[2] = { 1, 1 };
    double training_output_1[1] = { 0 };

    double training_input_2[2] = { 0, 0 };
    double training_output_2[1] = { 1 };

    // Train the Neural Network 1000000 times with the training data
    for( int i = 0; i < 1000000; i++ ) {
        // Switch between the two training sets
        if( i % 2 == 0 ) {
            train(nn, training_input_1, training_output_1);
        } else {
            train(nn, training_input_2, training_output_2);
        }
    }

    // Let the trained Neural Network predict the output of training set 1 (should be close to 0)
    double output[nn.output_nodes]; 
    predict(nn, training_input_1, output);
    printf("Output: %f\n", output[0]);

    // Destroy the Neural Network at the end
    destroyNeuralNetwork(nn);

    return 0;
}
```
## License
This project is licensed under the terms of the MIT license, see [LICENSE](LICENSE)