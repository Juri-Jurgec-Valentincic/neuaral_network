#include <cmath>
#include <fstream>
#include <chrono>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <cstdlib>
#include <cfloat>
#include <random>
#include "network.hpp"

using namespace Eigen;
using namespace std;

Matrix<double, Dynamic, 1> softmax(Matrix<double, Dynamic, 1> x){ // TODO optimize
        double  sum=0.0, m=-INFINITY, constarint;
        size_t n=x.rows();
        for(unsigned int i=0; i<n; i++)
                if(x(i) > m)
                        m = x(i);
      
        for(unsigned int i=0; i<n; i++)
                sum += exp(x(i) -m);
        constarint = m + log(sum);

        for(unsigned int i=0; i<n; i++)
                x(i) = exp(x(i) - constarint);
        return x;
}

void Layer::init_values(const unsigned int &previous_layer){    // normalized xavier initalization 4 weights
        double range = sqrt(6) / sqrt(previous_layer -  weight.rows() );
        mt19937_64 random_engine;
        uniform_real_distribution<double> distribution(-range, range);
        for(unsigned int i=0; i<weight.rows(); i++){
               for(unsigned int j=0; j<weight.cols(); j++){
                       weight(i,j) = distribution(random_engine);
               }
        }
        baias = Matrix<double, Dynamic, 1>::Zero(baias.rows());     // biase iniciliziram na 0
}

inline void Layer::forward_prop(){
        z =  weight * input;        // apply weight
        z = z.colwise() + baias;    // add bias
        a = z.unaryExpr(&sigmoid);      // activate   
}

inline void Output_layer::forward_prop(){
        z =  weight * input;
        z = z.colwise() + baias;
        for(unsigned int i=0; i<a.cols(); i++){
                a.col(i) = softmax(z.col(i));
        }
}

Layer::Layer(const size_t &neurons){
        baias.resize(neurons, NoChange);
        weight.resize(neurons, NoChange);
}

void Layer::set_input(const Matrix<double, Dynamic, Dynamic> &inp){
        input = inp; // maybe optimize
        if(first_time){
                weight.resize(NoChange, input.rows());
                a.resize(weight.rows(), input.cols());
                init_values(inp.rows());
                first_time=false;
        }
        else if(input.cols() != a.cols()){
                a.resize(NoChange, input.cols());
        }
        forward_prop();
}

void Network::create_network(const vector<unsigned int> &neurons_and_layers){
        cout << "creating network model...\t"<<flush;
        hiden_layers = neurons_and_layers.size();
        layers.reserve(neurons_and_layers.size()+2);
        layers.push_back(new Input_layer(INPUTS));
        for(unsigned int i=1;i<layers.capacity()-1;i++){
                layers.push_back(new Layer(neurons_and_layers[i-1]) );
        }
        layers.push_back(new Output_layer(10));
        cout << "done\n";
}


void Network::shufle(){ // fisher yates shufle
        shufled_dataset = *dataset;
        shufled_output  = *ideal_output;
        for(unsigned int i= SAMPLES-1; i>0; i--){
                unsigned int j = rand() % (i+1);
                Matrix<double,INPUTS,1>  temp = shufled_dataset.col(i);
                Matrix<double, OUTPUTS, 1> temp_ideal = shufled_output.col(i);
                shufled_dataset.col(i) = shufled_dataset.col(j);
                shufled_output.col(i) = shufled_output.col(j);
                shufled_output.col(j) = temp_ideal;
                shufled_dataset.col(j) = temp;
        }
}

Matrix<double, Dynamic, Dynamic> Network::deactivate(Matrix<double, Dynamic, Dynamic> input){
        for(unsigned int i=0; i<input.rows(); i++){
                for(unsigned int j=0; j<input.cols(); j++){
                        input(i, j) = sigmoid_derivetive(input(i,j));
                }
        }
        return input;
}

double sigmoid(const double &x){
        return 1/(1+exp(-x));
}

double sigmoid_derivetive(const double &x){
        double sigmoid_x = sigmoid(x);
        return sigmoid_x * (1-sigmoid_x);
}

double Network::calc_correctness(const out_matrix &actual_output, const out_matrix &ideal_output){
        unsigned int correct=0;
        out_matrix::Index a,b;
        for(unsigned int i=0; i<actual_output.cols(); i++){
                actual_output.col(i).maxCoeff(&a);
                ideal_output.col(i).maxCoeff(&b);
                if(a==b)
                        correct++;
        }
        return (double)correct / actual_output.cols();
}
void Network::run(const in_matrix *orignal_input, const out_matrix *perfect_output){
        cout << "starting training procedure" << endl;
        ideal_output = perfect_output;
        dataset = orignal_input;
        cout << "training for " << EPOCHS << " epochs, in batches of " << SAMPLES_IN_BATCH << endl;
        chrono::steady_clock::time_point start_start_time = chrono::steady_clock::now();
        for(unsigned int i=0; i<EPOCHS; i++){
                double correct_sum=0, correct_min=DBL_MAX, correct_max=DBL_MIN;
                cout << "shufifling dataset...\t" << flush;
                shufle();
                cout << "done\n";
                cout << "runing epoch: " << i << endl;
                chrono::steady_clock::time_point start_time = chrono::steady_clock::now();
                for(unsigned int j=0; j < SAMPLES / SAMPLES_IN_BATCH; j++){
                        ddm batch = shufled_dataset.block<INPUTS, SAMPLES_IN_BATCH>(0, j*SAMPLES_IN_BATCH);
                        ddm batch_ideal = shufled_output.block<OUTPUTS, SAMPLES_IN_BATCH>(0, j*SAMPLES_IN_BATCH);
                        forward_prop(batch);
                        double tmp = calc_correctness(layers.back()->a, batch_ideal);
                        correct_sum += tmp;
                        if(tmp > correct_max)
                                correct_max = tmp;
                        if(tmp < correct_min)
                                correct_min = tmp;
                        back_prop(batch_ideal);
                }
                chrono::duration<double> epoch_time = chrono::steady_clock::now() - start_time;
                cout << "epoch completed (" << epoch_time.count() << " sec)\n";
                cout << "estimated time remaining: " <<  epoch_time.count() * (EPOCHS - i) << " sec\n";
                cout << "min: " << correct_min*100 <<  "%"  << "\tmax: " << correct_max*100 << "%" << "\tavg: " << (correct_sum/(BATCHES_IN_EPOCH))*100 << "%\t(with testing dataset)" << endl << endl;
        }
        chrono::steady_clock::time_point end_time = chrono::steady_clock::now();
        cout << "training completed (" << EPOCHS << " epochs with "<< SAMPLES << " samples in batches of " << SAMPLES_IN_BATCH << ")\n";
        cout << "total training time: " << chrono::duration<double>(end_time - start_start_time).count() << " sec\t(" << chrono::duration<double>(end_time - start_start_time).count() / 60 << " min)\n";
}

double Network::test(const in_matrix &inp, const out_matrix &perfect_output){
        forward_prop(inp);
        return calc_correctness(layers.back()->a, perfect_output);
}

void Network::back_prop(const out_matrix &ideal_output){
        Matrix<double, Dynamic, Dynamic> delta_o = layers.back()->a - ideal_output; // -(ideal - actual_out ) == actual_out - ideal         napaka v outputu
        Matrix<double, Dynamic, Dynamic> delta_w = ( delta_o * layers.at(layers.size()-2)->a.transpose() ) / SAMPLES_IN_BATCH;           // derietive loss funkcije ( napaka * output prejsnga layerja.transpose ) / st samplov
        Matrix<double, Dynamic, Dynamic> delta_b = delta_o.rowwise().sum() / SAMPLES_IN_BATCH; // deriverive baiasa na output layerju 
        vector<Matrix<double, Dynamic, Dynamic>*> delta_weights;
        vector<Matrix<double, Dynamic, Dynamic>*> delta_baieses;
        delta_weights.reserve(hiden_layers);
        delta_baieses.reserve(hiden_layers);
        vector<Matrix<double, Dynamic, Dynamic>*>::iterator iw=delta_weights.begin(), ib=delta_baieses.begin();
        for(unsigned int i=hiden_layers; i>0; i--){
                *iw = new Matrix<double, Dynamic, Dynamic>;
                *ib = new Matrix<double, Dynamic, Dynamic>;
                delta_o = layers[i+1]->weight.transpose() * delta_o;
                delta_o = delta_o.array() * deactivate(layers[i]->z).array();
                **iw = ( delta_o * layers[i]->input.transpose() ) / SAMPLES_IN_BATCH;
                **ib = delta_o.rowwise().sum() / SAMPLES_IN_BATCH;
                ib++; iw++;
        }
        ib=delta_baieses.begin();
        iw=delta_weights.begin();
        layers.back()->weight -= LEARNING_RATE * delta_w;
        layers.back()->baias -= LEARNING_RATE * delta_b;
        for(unsigned int i=hiden_layers; i>0; i--){
                layers[i]->weight -= LEARNING_RATE * **iw;
                layers[i]->baias -= LEARNING_RATE * **ib;
                delete *iw;
                delete *ib;
                ib++; iw++;
        }
}

void Network::forward_prop(const in_matrix &input){
        Matrix<double, Dynamic, Dynamic> inp=input;
        for(vector<Layer*>::iterator it = layers.begin(); it != layers.end(); it++){
                (*it)->set_input(inp);
                inp = (*it)->a;
        }
}

void Network::save_network(const string &file){
        ofstream out_file(file, ofstream::binary);
        if(!out_file.is_open()){
                cerr << "couldn't open output file\texiting...\n";
                exit(1);
        }
        // wirite number of layers
        unsigned int tmp=layers.size();
        out_file.write((char*) &tmp, sizeof(tmp));
        // write number of neurons on layers 
        for(vector<Layer*>::iterator it=++layers.begin(); it!=layers.end()-1;it++){
                tmp = (*it)->baias.rows();
                out_file.write((char *)&tmp, sizeof(tmp));
        }
       // wite all wieghts 
        double temp;
        for(vector<Layer*>::iterator it=1+layers.begin(); it!=layers.end();it++){
                for(unsigned int i=0; i< (*it)->weight.rows(); i++){
                        for(unsigned int j=0; j < (*it)->weight.cols();j++){
                                temp = (*it)->weight(i, j);
                                out_file.write((char *) &temp, sizeof(temp));
                        }
                }
                for(unsigned int i=0; i< (*it)->baias.rows(); i++){
                        temp = (*it)->baias(i);
                        out_file.write((char *)&temp, sizeof(temp));
                }
        }
        char magic[] = "end-of-file";
        out_file.write(magic, 11);
        out_file.close();
}

/*
 *
 *-- file begin --
 32 uint num layers
 num layers * ( 32 uint num neurons)
 layer( doubles
        neuron * num neurons:
        weight
                row
                        collum
        baias
                row
) * num layers 
--file end --
*/
void Network::load_network(const string &file){
        ifstream in_file(file, ifstream::binary);
        if(!in_file.is_open()){
                cerr << "no such file or directory: " << file << endl << "exiting..." << endl;
                exit(1);
        }
        unsigned int tmp;
        in_file.read((char *)&tmp, sizeof(tmp)); // read number of layers
        vector<unsigned int> neurons_and_layers;
        neurons_and_layers.resize(tmp-2); // -input layer - output layer = -2
        for(vector<unsigned int>::iterator it=neurons_and_layers.begin(); it != neurons_and_layers.end();it++){
                in_file.read((char*) &(*it), sizeof(unsigned int) ); // read number of neurons on eatch layer
        }
        create_network(neurons_and_layers);

        // resize matricies to requierd size
        layers[1]->weight.resize(NoChange, INPUTS);
        layers[1]->first_time = false;
        layers[1]->a.resize(neurons_and_layers[0], NoChange);
        for(vector<Layer*>::iterator it=layers.begin()+1; it != layers.end(); it++){
               (*it)->weight.resize(NoChange, (*(it-1))->weight.rows() );
               (*it)->a.resize((*it)->weight.cols(), NoChange);
               (*it)->first_time = false;
        }
        layers.back()->a.resize(10, NoChange);

        double temp;
        for(vector<Layer*>::iterator it=layers.begin()+1; it != layers.end(); it++){
                for(unsigned int i=0; i< (*it)->weight.rows(); i++){ // iterate rows
                        for(unsigned int j=0;j< (*(it-1))->weight.rows(); j++){ // iterate colls
                                in_file.read((char*)&temp, sizeof(temp));
                                (*it)->weight(i,j) = temp;
                        }
                }
                for(unsigned int i=0; i< (*it)->baias.rows(); i++){
                        in_file.read((char*)&temp, sizeof(temp));
                        (*it)->baias(i) = temp;
                }
        }
        
        char magic[11];
        in_file.read(magic, 11);
        if(!strcmp(magic, "end-of-file")){
                cerr << "loading saved network failed\t" << magic << endl;
                in_file.close();
                exit(1);
        }
        else
                cout << "network loaded sucessfuly\n";
        in_file.close();
}

void Network::clasify(const in_matrix &inp){
        forward_prop(inp);
        int xd;
        layers.back()->a.col(0).maxCoeff(&xd);
        cout << "prediction: " << xd << "\tconfidence: " << layers.back()->a.col(0).maxCoeff()*100 << "%" << endl;
}

Network::~Network(){
        for(vector<Layer*>::iterator it = layers.begin(); it != layers.end(); it++){
                delete *it;
        }
}

// end-of-file
