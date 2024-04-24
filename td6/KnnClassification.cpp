#include "KnnClassification.hpp"
#include <iostream>
#include <ANN/ANN.h>

KnnClassification::KnnClassification(int k, Dataset *dataset, int col_class)
: Classification(dataset, col_class), m_k(k) 
{
    int d = dataset->get_dim() - 1;
    int n = dataset->get_n_samples();
    m_data_pts = annAllocPts(n, d);

    for (int i = 0; i < n; i++)
    {
        int idx = 0;
        for (int j = 0; j < dataset->get_dim(); j++) 
        {
            if (j != col_class)
            {
                m_data_pts[i][idx++] = dataset->get_instance(i)[j];
            }
        }
    }

    m_kd_tree = new ANNkd_tree(m_data_pts, n, d);
}

KnnClassification::~KnnClassification() {
    annDeallocPts(m_data_pts);
    delete m_kd_tree;
    annClose();
}
int KnnClassification::estimate(const ANNpoint &x, double threshold) const {
    ANNidxArray nnIdx = new ANNidx[m_k]; // Array for storing indices of nearest neighbors
    ANNdistArray dists = new ANNdist[m_k]; // Array for storing distances to nearest neighbors
    double eps = 0.0; // Error bound

    // Search for the k nearest neighbors of x
    m_kd_tree->annkSearch(x, m_k, nnIdx, dists, eps);

    double voteSum = 0; // This will accumulate the votes for the class

    // Sum up the votes for class 1 (assuming binary classification: 0 and 1)
    for (int i = 0; i < m_k; i++) {
        int pointIdx = nnIdx[i]; // Index of the ith nearest neighbor
        double label = m_dataset->get_instance(pointIdx)[m_col_class]; // Access the label of the nearest neighbor
        voteSum += label; // Add the label to the vote sum
    }

    // Calculate the average vote or probability
    double probability = voteSum / m_k;

    // Clean up
    delete[] nnIdx;
    delete[] dists;

    // Return 1 if the probability is greater than the threshold, otherwise 0
    return probability > threshold ? 1 : 0;
}


int KnnClassification::get_k() const 
{
    return m_k;
      }

ANNkd_tree *KnnClassification::get_kd_tree() {
    return m_kd_tree;
}

const ANNpointArray KnnClassification::get_points() const {
    return m_data_pts;}