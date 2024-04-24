#include "Dataset.hpp"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/SparseCore>

#ifndef RANDOMPROJECTION_HPP
#define RANDOMPROJECTION_HPP

/**
  The RandomProjection class .
*/
class RandomProjection {
    private:
        int m_original_dimension;
        int m_col_class;
        int m_projection_dim;
        std::string m_type_sample;
        Eigen::MatrixXd m_projection;
    public:
        /**
          A random Gaussian matrix (0, 1/projection_dim) of size (n, projection_dim).
          @param d original dimension
          @param projection_dim projection dimension (l in the TD)
        */
        static Eigen::MatrixXd random_gaussian_matrix(int d, int projection_dim);
        /**
          A random Gaussian matrix (0, 1/projection_dim) of size (n, projection_dim).
          @param d original dimension
          @param projection_dim projection dimension (l in the TD)
        */
        static Eigen::MatrixXi random_rademacher_matrix(int d, int projection_dim);
        /**
          The constructor.
          @param col_class the classification column
          @param projection_dim projection dimension (l in the TD)
          @param type_sample either "Gaussian" and anything else which would lead to a Rademacher random projection
        */
        RandomProjection(int original_dimension, int col_class, int projection_dim, std::string type_sample);
        /**
          Verify the quality of the projection as the mean distance between points in the original and projected data
        */
        void projection_quality(Dataset *dataset);
        /**
          Project dataset
          @param dataset
        */
        Dataset project(Dataset *dataset);   
        /**
          Original dimension getter
        */
        int get_original_dimension() const;
        /**
          Classification column getter
        */
        int get_col_class() const;
        /**
          Projection dimension getter
        */
        int get_projection_dim() const;
        /**
          Type of sampling getter
        */
        std::string get_type_sample() const;
        /**
          Projection matrix getter
        */
        Eigen::MatrixXd get_projection() const;
};

#endif    // END_RANDOMPROJECTION_HPP
