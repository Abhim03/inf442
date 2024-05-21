#pragma once

#include "node.hpp"

#include <iostream>
#include <functional>
#include <vector>

class Neuron
{
private:
    int nb_dendrites;
    std::vector<Node *> dendrites;
    Node *axon;

    // by default use the step function
    std::function<double(double)> activation;
    std::function<double(double)> activation_der;

    // by default use identity weights and no bias
    // vector should have length size + 1: [0] = bias
    std::vector<double> weights;

    // Collected input
    // used for both forward and backward propagation
    double collected_input = 0;
    double rate = 1;

    // For testing only
    static int count;
public:
    // Vanilla constructor
    Neuron(int _nb_dendrites);

    // Dendrites + activation function
    Neuron(int _nb_dendrites,
           std::vector<Node *> &_dendrites,
           std::function<double(double)> _activation,
           std::function<double(double)> _activation_der);
    Neuron(int _nb_dendrites,
           std::vector<Neuron *> &_neurons,
           std::function<double(double)> _activation,
           std::function<double(double)> _activation_der);

    ~Neuron();

    void init_weights();

    // Getters
    int get_nb_dendrites() const;
    const Node *get_dendrite(int pos) const;
    const Node *get_axon() const;
    const Node *get_bias_dendrite() const;
    double get_weight(int pos) const;
    double get_bias() const;
    double get_learning_rate() const;
    double get_output_signal() const;
    double get_collected_input() const; // for testing only
    static int get_count(); // for testing only

    // Setters
    void set_weights(std::vector<double> &_weights);
    void set_weight(int pos, double _weight);
    void set_bias(double _bias);
    void set_learning_rate(double _rate);
    void set_collected_input(double _collected_input); // for testing only

    void set_back_value(double back_value);

    // Forward propagation
    void step();
    // Backward propagation (training)
    void step_back();
};

std::ostream &operator<<(std::ostream &str, const Neuron &n);
