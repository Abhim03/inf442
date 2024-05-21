#include <iostream>
#include <fstream>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <algorithm>
#include <functional>
#include <vector>
#include <limits>
#include "dataset.hpp"
#include "perceptron.hpp"
#include "activation.hpp"

using namespace std;

void run_on_data(OneLayerPerceptron *perceptron,
                 Dataset &data,
                 int rounds,
                 int regr,
                 bool train = true,
                 bool print = false)
{
    clock_t tic, toc;
    clock_t cumulative = 0;
    double rss = 0;

    int rows = data.get_nb_samples();

    for (int round = 0; round < rounds; round++)
    {
        tic = clock();
        for (int row = 0; row < rows; row++)
        {
            if (print)
                cout << "Epoch/round = " << round
                     << ", row = " << row << endl;
            double output = perceptron->run(&data, row, regr, print);
            if (print)
                cout << "\tOutput: " << output;
            double err = (output - data.get_instance(row)[regr]);
            if (print)
                cout << "\tError: " << err << endl;
            rss += err * err;
            if (print)
                cout << "Current RSS: " << rss << endl;
        }

        if (train)
        {
            if (print)
                cout << "Updating learning rate...\t";
            perceptron->decay_learning_rate();
            if (print)
                cout << perceptron->get_learning_rate() << endl;
        }

        toc = clock();
        cumulative += toc - tic;
    }

    cout << "Mean RSS: " << rss / rounds << endl;
    cout << "Total time elapsed = "
         << cumulative / ((float)(CLOCKS_PER_SEC))
         << "s" << endl;
    cout << "Mean time per epoch/round = "
         << cumulative / ((float)(CLOCKS_PER_SEC)) / rounds
         << "s" << endl;
}

double evaluate_perceptron(OneLayerPerceptron *perceptron, Dataset &data, int regr)
{
    double rss = 0;
    int rows = data.get_nb_samples();

    for (int row = 0; row < rows; row++)
    {
        double output = perceptron->run(&data, row, regr, false);
        double err = (output - data.get_instance(row)[regr]);
        rss += err * err;
    }

    return rss / rows; // Mean RSS
}

int run(int argc, char *argv[])
{
    std::ifstream ftrain(argv[1]);
    std::ifstream ftest(argv[2]);

    if (ftrain.fail())
    {
        std::cout << "Cannot read from the training file" << std::endl;
        return 1;
    }

    if (ftest.fail())
    {
        std::cout << "Cannot read from the testing file" << std::endl;
        return 1;
    }

    Dataset training(ftrain);
    int dim = training.get_dim() - 1;
    int count = training.get_nb_samples();
    cout << "Read training data from " << argv[1] << endl;
    cout << count << " rows of dimension " << training.get_dim() << endl;

    Dataset testing(ftest);
    cout << "Read testing data from " << argv[2] << endl;
    cout << count << " rows of dimension " << testing.get_dim() << endl;

    assert(training.get_dim() == testing.get_dim());

    int regr = (argc > 3) ? std::atoi(argv[3]) : dim;
    int size = (argc > 4) ? std::atoi(argv[4]) : default_nb_neurons;
    int epochs = (argc > 5) ? std::atoi(argv[5]) : default_nb_epochs;
    double rate = (argc > 6) ? std::atof(argv[6]) : default_learning_rate;
    bool print = (argc > 7) ? std::atoi(argv[7]) == 1 : false;

    const int nb_trials = 10; // Number of trials with different initializations
    double best_performance = std::numeric_limits<double>::infinity();
    OneLayerPerceptron *best_perceptron = nullptr;

    for (int trial = 0; trial < nb_trials; ++trial)
    {
        OneLayerPerceptron *perceptron = new OneLayerPerceptron(dim, size, rate, rate / epochs, sigma, sigma_der);

        run_on_data(perceptron, training, epochs, regr, true, print);

        double performance = evaluate_perceptron(perceptron, testing, regr);

        if (performance < best_performance)
        {
            best_performance = performance;
            if (best_perceptron != nullptr)
            {
                delete best_perceptron;
            }
            best_perceptron = perceptron;
        }
        else
        {
            delete perceptron;
        }
    }

    if (best_perceptron != nullptr)
    {
        cout << "Best performance: " << best_performance << endl;

        cout << "Testing the perceptron on the training data (" << epochs << " times)" << endl;
        run_on_data(best_perceptron, training, epochs, regr, false, print);

        cout << "Testing the perceptron on the testing data (" << epochs << " times)" << endl;
        run_on_data(best_perceptron, testing, epochs, regr, false, print);

        cout << "Deleting the perceptron...\t";
        delete best_perceptron;
        cout << "done." << endl;
    }

    return 0;
}

int main(int argc, char *argv[])
{
    if (argc > 1)
    {
        if (argc < 3)
        {
            cout << "Usage: " << endl
                 << argv[0] << " <train_file> <test_file> " << endl
                 << "\t[ <column_for_regression> [ <number_of_neurons> [ <number_of_epochs> [ <learning_rate> [ <print> ] ] ] ] ] " << endl;
            return 1;
        }

        return run(argc, argv);
    }
}
