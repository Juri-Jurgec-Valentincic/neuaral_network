#ifndef LOAD_DATA_HPP
#define LOAD_DATA_HPP

#define HEIGHT 28
#define WIDTH 28
#define INPUTS HEIGHT*WIDTH
#define OUTPUTS 10
#define SAMPLES 60000

#include <fstream>
#include <string>
#include <array>
#include <vector>
#include <stdint.h>
#include <string>
#include <eigen3/Eigen/Dense>

using namespace std;

using namespace Eigen;




class Dataset{
        public:
                Dataset(const string file_name_training, const string file_name_training_lables, const string file_name_test, const string file_name_test_lables); // constructor validates datset files
                // load data from file an normalize it
                void load_training_data();
                void load_testring_data();
                // print an image
                static void print_image(const Matrix<double,1, INPUTS> &image);
                static unsigned char out_to_label(const Matrix<double,1,OUTPUTS> &network_output);
                // one-hot encodes labels for images
                void one_hot_encode(Matrix<double, OUTPUTS, Dynamic> &out,const vector<unsigned char>  &labels);

                // matrix of training and test data
                Matrix<double, INPUTS, Dynamic> training, test;
                Matrix<double, OUTPUTS, Dynamic> one_hot_train, one_hot_test;
                vector<unsigned char> test_lables, training_lables;
                private:
                // file objects for dataset files
                ifstream file_training, file_training_lables, file_test, file_test_lables;
                // lables of images in same order as in dataset files
                // number of training and testing images
                uint32_t test_images, training_images;
                // loads and normalizes data from dataset files
                void load_data(ifstream &data, ifstream &lables, Matrix<double,INPUTS, Dynamic> &out, vector<unsigned char> &labels_out, const uint32_t &number_of_images);
};


#endif
// end-of-file
