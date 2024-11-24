# A neural network for classifying images in the classic [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset
A lightweight neural network written in C++ using the [eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) library for matrix math. It uses fully connected layers, the sigmoid activation function and no convolutions. I achieved an accuracy of 97.89% on the testing dataset. I took on this project to learn about the inner workings of neural network.

# Building the project
The only dependency is [eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page).
On debian besed system it can be installed with:
```
sudo apt instal libeigen3-dev
```
To compile it just run:
```
make
```
I included the dataset and some pre trained models in the repo.

# Configuring, training and testing
The learning rate, samples in each training batch and number of epochs to train for can be found in [network.hpp](network.hpp)
The network architecture can be defined in [train.cpp](train.cpp)
To commit the changes in configuration recompile the project
To train the network run:
```
./train [model_output_file]
```
If no output file is provided it saves to "model.bin".

To test the trained network run:
```
./test [model_input_file]
```
If no file is provided it defaults to "models/model.bin".
