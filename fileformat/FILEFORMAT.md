# The file format for Neural Networks

## The files
[nn.bin](nn.bin) - the binary file, that contains the trained Neural Network from the example

[nn.txt](nn.txt) - the (kind of) human readable version of nn.bin

## The structure
1 byte: id, to check if this is the correct format, (42)
```
00101010
```
in C, it's the type `char`
***
8 bytes: the value of `NN_LEARNING_RATE`
```
00111111 10111001 10011001 10011001 10011001 10011001 10011001 10011010
```
in C, it's the type `double`
***
4 bytes: the value of `NN_ACTIVATION_FUNCTION`
```
00000000 00000000 00000000 00000000
```
in C, it's the type `int`
***
4 bytes: the amount of input nodes
```
00000010 00000000 00000000 00000000
```
in C, it's the type `int`
***
4 bytes: the amount of hidden nodes
```
00000001 00000000 00000000 00000000
```
in C, it's the type `int`
***
4 bytes: the amount of output nodes
```
00000001 00000000 00000000 00000000
```
in C, it's the type `int`
***
(8 * `input nodes` * `hidden nodes`) bytes: the data of the weights between input and hidden nodes
```
01010110 11101010 01101000 11001011 11111010 01011010 00100000 01000000 01010001 01100000 01001111 11011000 01010010 01111100 00011111 01000000
```
in C, it's the type `double*`
***
(8 * `hidden nodes` * `output nodes`) bytes: the data of the weights between hidden and output nodes
```
10100001 10110111 01011010 00111111 10001001 00110010 00100101 11000000 
```
in C, it's the type `double*`
***
(8 * `hidden nodes`) bytes: the data of the bias of the hidden nodes
```
11101011 11000010 01101111 01011101 00001000 00011110 00011111 11000000
```
in C, it's the type `double*`
***
(8 * `output nodes`) bytes: the data of the bias of the output nodes
```
10110101 00010110 00000010 10110111 11101101 11001010 00010100 01000000
```
in C, it's the type `double*`