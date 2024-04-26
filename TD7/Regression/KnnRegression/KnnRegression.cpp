#include <iostream>
#include <ANN/ANN.h>
#include "KnnRegression.hpp"






KnnRegression::KnnRegression(int k, Dataset* dataset, int col_regr)
: Regression(dataset, col_regr), m_k(k) {
    int n = dataset->get_nbr_samples();  // Get the number of samples
    int d = dataset->get_dim() - 1;      // Dimensionality minus the regression column

    m_dataPts = annAllocPts(n, d);       // Allocate space for points

    for (int i = 0; i < n; i++) {
        const std::vector<double>& instance = dataset->get_instance(i);
        int index = 0;  // Index for the m_dataPts array

        for (int j = 0; j < instance.size(); j++) {
            if (j != col_regr) {
                // Assign the point, excluding the regression column
                m_dataPts[i][index++] = instance[j];
            }
        }
    }

    // Create the k-d tree using the ANN library
    m_kdTree = new ANNkd_tree(m_dataPts, n, d);
}

KnnRegression::~KnnRegression() {
    if (m_kdTree) {
        delete m_kdTree;
    }
    annDeallocPts(m_dataPts);  // Deallocate the points array
    annClose();                // Clean up the ANN library
}







double KnnRegression::estimate(const Eigen::VectorXd &x) const {
    assert(x.size() == m_dataset->get_dim() - 1);
    
    ANNidxArray nnIdx = new ANNidx[m_k];      // Allocate near neighbor indices
    ANNdistArray dists = new ANNdist[m_k];    // Allocate near neighbor distances
    ANNpoint queryPt = annAllocPt(x.size());  // Allocate query point

    // Copy data from Eigen vector to ANNpoint
    std::copy(x.data(), x.data() + x.size(), queryPt);

    // Search for the nearest neighbors
    m_kdTree->annkSearch(queryPt, m_k, nnIdx, dists);

    double sum_y = 0.0;
    for (int i = 0; i < m_k; i++) {
        const std::vector<double>& nn_instance = m_dataset->get_instance(nnIdx[i]);
        sum_y += nn_instance[m_col_regr];
    }

    double predicted_y = sum_y / m_k;  // Compute the mean of the Y-components

    // Clean up
    delete[] nnIdx;
    delete[] dists;
    annDeallocPt(queryPt);

    return predicted_y;
}


int KnnRegression::get_k() const {
	return m_k;
}

ANNkd_tree* KnnRegression::get_kdTree() const {
	return m_kdTree;
}
