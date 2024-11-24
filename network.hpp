#ifndef NETWORK_HPP
#define NETWORK_HPP

#ifndef LEARNING_RATE
#define LEARNING_RATE 0.01
#endif

#ifndef SAMPLES_IN_BATCH
#define SAMPLES_IN_BATCH 30
#endif

#ifndef EPOCHS
#define EPOCHS 10
#endif 

#if SAMPLES % SAMPLES_IN_BATCH != 0
        #warn "all samples in dataset can't be evenly divided into batches"
#endif

#define BATCHES_IN_EPOCH (float)SAMPLES / SAMPLES_IN_BATCH

#include <eigen3/Eigen/Dense>
#include <vector>
#include <string>
#include "load_data.hpp"

using namespace std;
using namespace Eigen;

typedef Matrix<double, OUTPUTS, Dynamic> out_matrix;
typedef Matrix<double, INPUTS, Dynamic> in_matrix;
typedef Matrix<double, Dynamic, Dynamic> double_dynamic_matrix;
typedef double_dynamic_matrix ddm; 

double sigmoid(const double &x);
double sigmoid_derivetive(const double &x);
Matrix<double, Dynamic, 1> softmax(Matrix<double, Dynamic, 1> x);

class Layer{
        public:
                Layer(const size_t &neurons);
                Layer(const size_t &neurons, const size_t &previous_layer);
                virtual void set_input(const Matrix<double, Dynamic, Dynamic> &inp);
                Matrix<double, Dynamic, Dynamic> weight, z, a, input; 
                Matrix<double, Dynamic, 1> baias;
                bool first_time=true;
        protected:
                void init_values(const unsigned int &previous_layer);
                virtual void forward_prop();
};

class Output_layer: public Layer{
        public:
                Output_layer(const size_t &neurons): Layer(neurons){}
        protected:
                void forward_prop();
};

class Input_layer: public Layer{
        public:
                Input_layer(const size_t &neurons): Layer(neurons){}
                void set_input(const Matrix<double, Dynamic, Dynamic> &inp){
                        a=inp;
                        input=inp;
                }

};

class Network{
        public:
               void create_network(const vector<unsigned int> &neurons_and_layers);
               void run(const in_matrix *inp, const out_matrix *perfect_output);
               double test(const in_matrix &inp, const out_matrix &perfect_output);
               void save_network(const string &file="models/model.bin");
               void clasify(const in_matrix &inp);
               void load_network(const string &file);
               Network(){}
               ~Network();
               Network(const vector<unsigned int> &net){ create_network(net); }
               Network(const string &file) { load_network(file); }
        private:
               vector<Layer*> layers;
               const out_matrix *ideal_output;
               const in_matrix *dataset;
               unsigned int hiden_layers;
               out_matrix shufled_output;
               in_matrix shufled_dataset;
               void back_prop(const out_matrix &ideal_output);
               void forward_prop(const in_matrix &input);
               void shufle();
               double calc_correctness(const out_matrix &actual_output, const out_matrix &ideal_output);
               Matrix<double, Dynamic, Dynamic> deactivate( Matrix<double, Dynamic, Dynamic> input);
};

#endif
// end-of-file


