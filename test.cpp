#include <iostream>
#include <string>
#include <filesystem>
#include "network.hpp"
#include "load_data.hpp"

#include <ctime>

using namespace std;
string save_file("models/model.bin");

int main(int argc, char *argv[]){
        srand(time(NULL));
        if(argc == 1){
                cout << "no file specified defaulting to models/model.bin\n";
        }
        else if( (argc > 1 && ( !strcmp(argv[1], "-h") || !strcmp(argv[1], "--help") || !strcmp(argv[1], "-help") )) || argc > 2 ) {
                cout << "usage: .test [file] \n if no file specified defaults to \"model.bin\" \n";
                return 0;
        }
        else if(argc == 2){
                cout << "loading network froms: " << argv[1] << endl;
        }
        filesystem::current_path(filesystem::read_symlink("/proc/self/exe").parent_path()); // change ralative path
        Dataset mnist("dataset/train-images-idx3-ubyte", "dataset/train-labels-idx1-ubyte", "dataset/t10k-images-idx3-ubyte", "dataset/t10k-labels-idx1-ubyte");
        mnist.load_testring_data();
        Network net;
        cout << "loading saved model... \n";
        net.load_network(save_file);
        cout << "testing model\n";
        cout << "accuracy with testing dataset: " << net.test(mnist.test, mnist.one_hot_test)*100 << "%" << endl ;
        for(int i=100;i<100;i++)
        {
                int neki = rand() % 10000;
                Dataset::print_image(mnist.test.col(neki));
                net.clasify(mnist.test.col(neki));
                cout << "label: " << (int)mnist.test_lables[neki]<< endl;
        }
        while(1){
                string num;
                cout << "image to classify (0 - 9999): ";
                getline(cin, num);
                int tmp;
                try{
                        tmp = stoi(num, nullptr, 10);
                }
                catch (const std::invalid_argument& ia){
                        cout << "invalid input\n";
                        continue;
                }
                if(tmp > 9999 || tmp < 0){
                        cout << "invalid input\n";
                        continue;
                }
                Dataset::print_image(mnist.test.col(tmp));
                cout << "label: " << (int)mnist.test_lables[tmp]<< endl;
                net.clasify(mnist.test.col(tmp));
        }
        return 0;
}

