#ifndef __gnu_linux__
        #warn "os is not GNU/linux porbably won't work"
#endif
#include <iostream>
#include <ctime>
#include <cstdlib>
#include <vector>
#include <filesystem>
#include "load_data.hpp"
#include "network.hpp"
#include <cstring>

using namespace std;

int main(int argc, char *argv[]){
        if(argc > 1 && ( !strcmp(argv[1], "-h") || !strcmp(argv[1], "--help") || !strcmp(argv[1], "-help") ) ) {
                cout << "usage: ./train [out_file] \n if no file specified defaults to \"model.bin\" \n";
                return 0;
        }
        filesystem::current_path(filesystem::read_symlink("/proc/self/exe").parent_path()); // change ralative path
        srand(time(NULL));
        // load dataset
        Dataset mnist("dataset/train-images-idx3-ubyte", "dataset/train-labels-idx1-ubyte", "dataset/t10k-images-idx3-ubyte", "dataset/t10k-labels-idx1-ubyte");
        mnist.load_training_data();
        mnist.load_testring_data();
        // declare network architecture
        vector<unsigned int> network;
        network.push_back(700);
        network.push_back(500);
        network.push_back(300);
        network.push_back(100);
        Network net;
        net.create_network(network);
        // train network
        net.run(&mnist.training, &mnist.one_hot_train);
        net.save_network();
        cout << "acuracy with testing dataset: " << net.test(mnist.test, mnist.one_hot_test)*100 << "%" << endl ;
        return 0;
}
