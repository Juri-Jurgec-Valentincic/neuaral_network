#include <fstream>
#include <string>
#include <iostream>
#include <stdint.h>
#include <arpa/inet.h>
#include <limits.h>
#include <eigen3/Eigen/Dense>
#include "load_data.hpp"
#include <unistd.h>

using namespace std;
using namespace Eigen;

Dataset::Dataset(const string file_name_training, const string file_name_training_lables, const string file_name_test, const string file_name_test_lables){
        ifstream *files[] = {&file_training, &file_training_lables, &file_test, &file_test_lables};
        const string *file_names[] = {&file_name_training, &file_name_training_lables, &file_name_test, &file_name_test_lables};
        
        for(unsigned char i=0; i<4; i++){
                files[i]->open(*file_names[i], ios::in | ios::binary);
                if(!files[i]->is_open()){
                        cerr << *file_names[i] <<  ": no such file or directory\n";
                        exit(1);
                }
        }


        for(unsigned char i=0; i<4; i++){
                uint32_t magic_number;
                files[i]->read((char *)&magic_number, sizeof(magic_number));
                magic_number = ntohl(magic_number);
                if(magic_number != 0x801 && magic_number != 0x803){
                        cerr << *file_names[i] << " dosent have the correct magic number ( invalid IDX file )\n";
                        exit(1);
                }
        }

        file_training.read( (char*) &training_images, sizeof(training_images));
        training_images = ntohl(training_images);
        file_test.read( (char*) &test_images, sizeof(test_images) );
        test_images = ntohl(test_images);
        uint32_t tmp;
        file_training_lables.read( (char*) &tmp, sizeof(tmp) );
        tmp = ntohl(tmp);
        if(tmp != training_images ){
               cerr << "not same amount of lables and images in training dataset\n";
               exit(1);
        }
        file_test_lables.read( (char*) &tmp, sizeof(tmp) );
        tmp = ntohl(tmp);
        if(tmp != test_images ){
               cerr << "not same amount of lables and images in testing dataset\n";
               exit(1);
        }

        for(unsigned char i=0; i<2;i++){
                file_training.read((char*) &tmp, sizeof(tmp));
                tmp = ntohl(tmp);
                if(tmp !=28){
                        cerr << "height or with of image in: " << file_name_training << endl;
                        exit(1);
                }
        }

        for(unsigned char i=0; i<2;i++){
                file_test.read((char*) &tmp, sizeof(tmp));
                tmp = ntohl(tmp);
                if(tmp !=28){
                        cerr << "height or with of image in: " << file_name_test << endl;
                        exit(1);
                }
        }
        cout << "found dataset...\n";
}


void Dataset::load_data(ifstream &data, ifstream &lables, Matrix<double, INPUTS, Dynamic> &out, vector<unsigned char> &labels_out, const uint32_t &number_of_images){
        out.resize(NoChange, number_of_images);
        labels_out.resize(number_of_images, UCHAR_MAX);
        array <unsigned char, INPUTS>  tmp;
        for(uint32_t i=0; i<number_of_images; i++){
                data.read( (char*) tmp.data(), sizeof(unsigned char)*HEIGHT*WIDTH);
                lables.read( (char*) &labels_out.at(i), sizeof(unsigned char));
                for(unsigned short j=0; j<INPUTS; j++){
                        out(j,i) = (double) tmp[j] / 255;
                }
        }
}

void Dataset::load_training_data(){
        cout << "loading into memory, normalizing and one-hot encodeing training data...\t" << flush;
        load_data(file_training, file_training_lables, training, training_lables, training_images);
        one_hot_encode(one_hot_train, training_lables);
        cout << "done\n";
}

void Dataset::load_testring_data(){
        cout << "loading into memory, normalizing and one-hot encodeing testing  data...\t" << flush;
        load_data(file_test, file_test_lables, test, test_lables, test_images);
        one_hot_encode(one_hot_test, test_lables);
        cout << "done\n";
}

void Dataset::print_image(const Matrix<double, 1, INPUTS> &image){
        for(unsigned char i=0; i<HEIGHT; i++){
                for(unsigned char j=0; j<WIDTH; j++){
                        if(image(i*WIDTH+j) == 0)
                                cout << ' ';
                        else{
                                unsigned short rgb=image(i*WIDTH+j)*255;
                                string color("\033[38;2;");
                                for(unsigned char h=0;h<3;h++)
                                        //color.append(to_string(rand() % 230 + 20) + ';' );
                                        color.append(to_string(rgb)+';');
                                color.erase(color.size()-1);
                                color.append("m");
                                cout << color << "\u2588";
                        }
                }
                cout << endl;
        }
        cout << "\033[0m" << flush;
}


void Dataset::one_hot_encode(Matrix<double, OUTPUTS, Dynamic> &out,const vector<unsigned char> &labels){
        out.resize(NoChange, labels.size());
        for(unsigned int j=0; j<out.cols(); j++){
                for(unsigned char i=0; i<10; i++){
                        if(i==labels[j])
                                out(i,j) = 1.0;
                        else
                                out(i,j) = 0;
                }
        }
}

unsigned char Dataset::out_to_label(const Matrix<double, 1,OUTPUTS> &network_output){
        unsigned char xd;
        network_output.maxCoeff(&xd);
        return xd;

}

// end-of-file 
