#pragma once

#include "dataset.hpp"
#include "neuron.hpp"

const int default_nb_neurons = 5;
const int default_nb_epochs = 100;
const double default_learning_rate = 0.1;

class OneLayerPerceptron
{
protected:
    int dim;
    int size;
    std::vector<Node *> inputs;
    std::vector<Neuron *> hidden;
    Neuron *output;

    int epoch;
    double rate;
    double decay;

public:
    OneLayerPerceptron(int _dim, int _size, double _rate, double _decay,
                       std::function<double(double)> _activation,
                       std::function<double(double)> _activation_der);
    ~OneLayerPerceptron();

    int get_nb_neurons();

    double get_learning_rate();
    double get_decay();

    // "Using Learning Rate Schedules for Deep Learning Models
    // in Python with Keras", by Jason Brownlee
    // https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/
    void set_learning_rate(double _rate);
    void init_learning_rate(double _rate);
    void decay_learning_rate();

protected:
    double normalize(double val, Dataset *data, int coord);
    double denormalize(double val, Dataset *data, int coord);

    virtual void prepare_inputs(Dataset *data, int row, int regr, bool print = false);
    virtual void compute_hidden_step(bool print = false);
    virtual double compute_output_step(Dataset *data, int row, int regr, bool print = false);
    virtual void propagate_back_hidden(bool print = false);

public:
    virtual double run(Dataset *data, int row, int regr, bool print = false);
};
