#include<iostream>
#include<cassert>
#include "LinearRegression.hpp"
#include "Dataset.hpp"
#include "Regression.hpp"

LinearRegression::LinearRegression(Dataset* dataset, int col_regr) 
: Regression(dataset, col_regr) {
	m_beta = nullptr;
	set_coefficients();
}

LinearRegression::~LinearRegression() {
	if (m_beta != nullptr) {
		m_beta->resize(0);
		delete m_beta;
	}
}




namespace {
    std::pair<Eigen::MatrixXd, Eigen::VectorXd> construct_out(const Dataset* dataset, int col_regr) {
        int n = dataset->get_nbr_samples();
        int p = dataset->get_dim();

        Eigen::MatrixXd X(n, p); 
        Eigen::VectorXd y(n);

        X.col(0) = Eigen::VectorXd::Ones(n); // Colonne pour l'intercept
        for (int i = 0; i < n; ++i) {
            y(i) = dataset->get_instance(i)[col_regr];
            int j = 1;
            for (int k = 0; k < p; ++k) {
                if (k != col_regr) {
                    X(i, j) = dataset->get_instance(i)[k];
                    j++;
                }
            }
        }
        return {X, y};
    }
}


Eigen::MatrixXd LinearRegression::construct_matrix() {
    auto result = construct_out(m_dataset, m_col_regr);  // Ajoutez m_col_regr
    return result.first;
}

Eigen::VectorXd LinearRegression::construct_y() {
    auto result = construct_out(m_dataset, m_col_regr);  // Ajoutez m_col_regr
    return result.second;
}





void LinearRegression::set_coefficients() {
    
    Eigen::MatrixXd X = this->construct_matrix();
    Eigen::VectorXd y = this->construct_y();

    
    m_beta = new Eigen::VectorXd(m_dataset->get_dim());


    *m_beta = (X.transpose() * X).ldlt().solve(X.transpose() * y);
}





const Eigen::VectorXd* LinearRegression::get_coefficients() const {
	if (!m_beta) {
		std::cout <<"Coefficients have not been allocated." <<std::endl;
		return NULL;
	}
	return m_beta;
}

void LinearRegression::show_coefficients() const {
	if (!m_beta) {
		std::cout << "Coefficients have not been allocated." <<std::endl;
		return;
	}
	
	if (m_beta->size() != m_dataset->get_dim()) {  // ( beta_0 beta_1 ... beta_{d} )
		std::cout << "Warning, unexpected size of coefficients vector: " << m_beta->size() << std::endl;
	}
	
	std::cout<< "beta = (";
	for (int i=0; i<m_beta->size(); i++) {
		std::cout << " " << (*m_beta)[i];
	}
	std::cout << " )" <<std::endl;
}

void LinearRegression::print_raw_coefficients() const {
	std::cout<< "{ ";
	for (int i = 0; i < m_beta->size() - 1; i++) {
		std::cout << (*m_beta)[i] << ", ";
	}
	std::cout << (*m_beta)[m_beta->size() - 1];
	std::cout << " }" << std::endl;
}


// Correction de la signature de sum_of_squares pour correspondre à l'en-tête
void LinearRegression::sum_of_squares(Dataset* dataset, double& ess, double& rss, double& tss) const {
    auto result = construct_out(dataset, m_col_regr);  // Assurez-vous d'ajouter m_col_regr si nécessaire
    Eigen::MatrixXd X = result.first;
    Eigen::VectorXd y = result.second;

    Eigen::VectorXd predictions = X * (*m_beta);
    double empirical_mean = 0;

    for (int i = 0; i < dataset->get_nbr_samples(); i++) {
        empirical_mean += y[i];
    }
    empirical_mean /= (double)(dataset->get_nbr_samples());

    ess = rss = tss = 0;
    for (int i = 0; i < dataset->get_nbr_samples(); i++) {
        tss += std::pow(y[i] - empirical_mean, 2);
        ess += std::pow(predictions[i] - empirical_mean, 2);
        rss += std::pow(predictions[i] - y[i], 2);
    }
}




double LinearRegression::estimate(const Eigen::VectorXd & x) const {
    Eigen::VectorXd augmented_x(x.size() + 1);
    augmented_x << 1, x; // Prepend a 1 for the intercept term
    return (*m_beta).dot(augmented_x); // Use dot product
}

