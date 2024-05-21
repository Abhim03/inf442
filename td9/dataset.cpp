#include "dataset.hpp"

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <limits>
#include <cassert>

int Dataset::get_nb_samples() const
{
    return nsamples;
}

int Dataset::get_dim() const
{
    return dim;
}

Dataset::~Dataset()
{
}

void Dataset::show(bool verbose)
{
    std::cout << "Dataset with "
              << nsamples << " samples, and "
              << dim << " dimensions." << std::endl;

    if (verbose)
    {
        for (int i = 0; i < nsamples; i++)
        {
            for (int j = 0; j < dim; j++)
            {
                std::cout << instances[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }
}

Dataset::Dataset(std::ifstream &fin) : min(0), max(0)
{
    nsamples = 0;
    dim = -1;

    std::vector<double> row;
    std::string line, word, temp;

    while (getline(fin, line))
    {
        row.clear();
        std::stringstream s(line);

        int valid_sample = 1;
        int ncols = 0;
        while (getline(s, word, ','))
        {
            // add all the column data
            // of a row to a vector

            double val = std::atof(word.c_str());
            row.push_back(val);
            ncols++;
        }
        if (!valid_sample)
        {
            // in this version, valid_sample is always 1
            continue;
        }
        instances.push_back(row);
        if (dim == -1)
            dim = ncols;
        else if (dim != ncols)
        {
            std::cerr << "ERROR, inconsistent dataset" << std::endl;
            exit(-1);
        }

        nsamples++;
    }

    fin.close();

    for (int i = 0; i < dim; i++)
    {
        max.push_back(std::numeric_limits<double>::lowest());
        min.push_back(std::numeric_limits<double>::max());
        mean.push_back(0);
        variance.push_back(0);
        std_dev.push_back(0);
    }
}

std::vector<double> &Dataset::get_instance(int i)
{
    return instances[i];
}

std::vector<double> *Dataset::get_mins()
{
    if (min[0] == std::numeric_limits<double>::max())
        compute_stats();

    return &min;
}

std::vector<double> *Dataset::get_maxs()
{
    if (max[0] == std::numeric_limits<double>::min())
        compute_stats();

    return &max;
}

double Dataset::get_min(int i) {
    assert(0 <= i && i < dim);

    if (min[0] == std::numeric_limits<double>::max())
        compute_stats();

    return min[i];
}

double Dataset::get_max(int i) {
    assert(0 <= i && i < dim);

    if (max[0] == std::numeric_limits<double>::lowest())
        compute_stats();

    return max[i];
}

double Dataset::get_mean(int i) {
    assert(0 <= i && i < dim);

    if (mean[0] == 0)
        compute_stats();

    return mean[i];
}

double Dataset::get_variance(int i) {
    assert(0 <= i && i < dim);

    if (variance[0] == 0)
        compute_stats();

    return variance[i];
}

double Dataset::get_std_dev(int i) {
    assert(0 <= i && i < dim);

    if (std_dev[0] == 0)
        compute_stats();

    return std_dev[i];
}

void Dataset::compute_stats()
{
    for (auto row : instances)
    {
        for (int i = 0; i < dim; i++)
        {
            if (row[i] > max[i])
                max[i] = row[i];
            if (row[i] < min[i])
                min[i] = row[i];

            // For our purposes, we assume that they will not overflow
            mean[i] += row[i];
            variance[i] += row[i]*row[i];
        }
    }

    for (int i = 0; i < dim; i++) {
        mean[i] /= instances.size();
        variance[i] /= instances.size();
        variance[i] -= mean[i]*mean[i];
        std_dev[i] = sqrt(variance[i]);
    }
}

void Dataset::merge_stats (Dataset &that) {
    if (max[0] == std::numeric_limits<double>::min())
        compute_stats();

    if (that.max[0] == std::numeric_limits<double>::min())
        that.compute_stats();

    for (int i = 0; i < dim; i++) {
        min[i] = std::min(min[i], that.min[i]);
        that.min[i] = min[i];
        max[i] = std::max(max[i], that.max[i]);
        that.max[i] = max[i];

        variance[i] += mean[i]*mean[i];
        variance[i] *= instances.size();
        that.variance[i] += that.mean[i]*that.mean[i];
        that.variance[i] *= that.instances.size();

        mean[i] = (mean[i]*instances.size() + that.mean[i]*that.instances.size())/(instances.size() + that.instances.size());

        variance[i] += that.variance[i];
        variance[i] /= instances.size() + that.instances.size();
        variance[i] -= mean[i]*mean[i];

        std_dev[i] = sqrt(variance[i]);

        that.mean[i] = mean[i];
        that.variance[i] = variance[i];
        that.std_dev[i] = std_dev[i];
    }
}

std::ostream &operator<<(std::ostream &str, const std::vector<double> &vx)
{
    for (int i = 0; i < vx.size(); i++)
        str << vx[i] << "\t";
    return str;
}
