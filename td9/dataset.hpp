#pragma once

#include <iostream>
#include <vector>
#include <cstdlib>
#include <string>

class Dataset
{
public:
    Dataset(std::ifstream &fin);
    ~Dataset();

    void show(bool verbose);
    std::vector<double> &get_instance(int i);
    int get_nb_samples() const;
    int get_dim() const;

    std::vector<double> *get_mins();
    std::vector<double> *get_maxs();
    void merge_stats (Dataset &that);

    double get_min(int i);
    double get_max(int i);
    double get_mean(int i);
    double get_variance(int i);
    double get_std_dev(int i);

private:
    int dim;
    int nsamples;
    std::vector<std::vector<double> > instances;

    std::vector<double> min;
    std::vector<double> max;
    std::vector<double> mean;
    std::vector<double> variance;
    std::vector<double> std_dev;

    void compute_stats();
};

std::ostream &operator<<(std::ostream &str, const std::vector<double> &vx);
