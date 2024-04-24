#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstdarg>
#include <iterator>
#include <string>
#include <regex>
#include <numeric>
#include <cmath>

#include <ANN/ANN.h>
#include <ANN/ANNperf.h>

#include "../gradinglib/gradinglib.hpp"
#include "../Classification.hpp"
#include "../Dataset.hpp"
#include "../ConfusionMatrix.hpp"
#include "../KnnClassification.hpp"
#include "../RandomProjection.hpp"
#include <limits>

namespace tdgrading {

using namespace testlib;
using namespace std;


int test_q1(std::ostream &out, const std::string test_name) {
    std::string entity_name = "knn_class";
    start_test_suite(out, test_name);

    // testing params
    const char* dataset_file = "csv/mail_train.csv";
    int k = 3;
    int col_class = 0;

    Dataset train_dataset(dataset_file);
    KnnClassification knn_class(k, &train_dataset, col_class);
    ANNkdStats stats;
    knn_class.get_kd_tree()->getStats(stats);

    std::vector<int> res = {
      test_eq(out, "data loaded", knn_class.get_points()[25][1898], 1),
      test_eq(out, "data loaded", knn_class.get_points()[28][4], 1),
      test_eq(out, "data loaded", knn_class.get_points()[3999][1], 0),
      test_eq(out, "data loaded", knn_class.get_points()[3999][5], 1),
      test_eq(out, "getColClass", knn_class.get_col_class(), col_class),
      test_eq(out, "getK", knn_class.get_k(), k),
      test_eq(out, "dim", stats.dim, 1899),
      test_eq(out, "n_pts", stats.n_pts, 4000),
      test_eq(out, "depth", stats.depth, 203)
    };

    return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}

//-------------------------------------------------------------------

int test_q2(std::ostream &out, const std::string test_name) {
    std::string entity_name = "knn_class";
    start_test_suite(out, test_name);

    // testing params
    const char* dataset_train_file = "csv/mail_train.csv";
    const char* dataset_test_file = "csv/mail_test.csv";
    int k = 3;
    int col_class = 0;

    // Puts train and test files in a Dataset object
	Dataset train_dataset(dataset_train_file);
	Dataset class_dataset(dataset_test_file);
    KnnClassification knn_class(k, &train_dataset, col_class);

	double mer = 0;
	for (int i = 0; i < class_dataset.get_n_samples(); i++) {
		std::vector<double> sample = class_dataset.get_instance(i);
		// extract column for classification
	    ANNpoint query = annAllocPt(class_dataset.get_dim() - 1, 0);
		for (int j = 0, j2 = 0; j < train_dataset.get_dim() - 1 && j2 < train_dataset.get_dim(); j++, j2++) {
			if (j == col_class && j2 == col_class) {
				j--;
				continue;
			}
			query[j] = sample[j2];
		}
		double estim = knn_class.estimate(query);
		annDeallocPt(query);
		mer += abs(estim - sample[col_class]) / class_dataset.get_n_samples();
	}

    std::vector<int> res = {
      test_le(out, "Error_rate", mer, 0.104)
    };

    return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}

//-------------------------------------------------------------------

int test_q3(std::ostream &out, const std::string test_name) {
    std::string entity_name = "ConfusionMatrix";
    start_test_suite(out, test_name);

    std::vector<int> res;

    // ConfusionMatrix
    ConfusionMatrix confusion_matrix;

    res.push_back(test_eq(out, "get_fn", confusion_matrix.get_fn(), 0));
    res.push_back(test_eq(out, "get_fp", confusion_matrix.get_fp(), 0));
    res.push_back(test_eq(out, "get_tn", confusion_matrix.get_tn(), 0));
    res.push_back(test_eq(out, "get_tp", confusion_matrix.get_tp(), 0));


    // Test good entry
    confusion_matrix.add_prediction(0,0);
    res.push_back(test_eq(out, "get_fn", confusion_matrix.get_fn(), 0));
    res.push_back(test_eq(out, "get_fp", confusion_matrix.get_fp(), 0));
    res.push_back(test_eq(out, "get_tn", confusion_matrix.get_tn(), 1));
    res.push_back(test_eq(out, "get_tp", confusion_matrix.get_tp(), 0));

    // Test good entry
    confusion_matrix.add_prediction(1,1);
    res.push_back(test_eq(out, "get_fn", confusion_matrix.get_fn(), 0));
    res.push_back(test_eq(out, "get_fp", confusion_matrix.get_fp(), 0));
    res.push_back(test_eq(out, "get_tn", confusion_matrix.get_tn(), 1));
    res.push_back(test_eq(out, "get_tp", confusion_matrix.get_tp(), 1));

    // Test good entry
    confusion_matrix.add_prediction(0,1);
    res.push_back(test_eq(out, "get_fn", confusion_matrix.get_fn(), 0));
    res.push_back(test_eq(out, "get_fp", confusion_matrix.get_fp(), 1));
    res.push_back(test_eq(out, "get_tn", confusion_matrix.get_tn(), 1));
    res.push_back(test_eq(out, "get_tp", confusion_matrix.get_tp(), 1));

    // Test good entry
    confusion_matrix.add_prediction(1,0);
    res.push_back(test_eq(out, "get_fn", confusion_matrix.get_fn(), 1));
    res.push_back(test_eq(out, "get_fp", confusion_matrix.get_fp(), 1));
    res.push_back(test_eq(out, "get_tn", confusion_matrix.get_tn(), 1));
    res.push_back(test_eq(out, "get_tp", confusion_matrix.get_tp(), 1));

    // Test metrics
    res.push_back(test_eq(out, "precision", confusion_matrix.precision(), 0.5));
    res.push_back(test_eq(out, "f_score", confusion_matrix.f_score(), 0.5));
    res.push_back(test_eq(out, "false_alarm_rate", confusion_matrix.false_alarm_rate(), 0.5));
    res.push_back(test_eq(out, "detection_rate", confusion_matrix.detection_rate(), 0.5));
    res.push_back(test_eq(out, "error_rate", confusion_matrix.error_rate(), 0.5));

    return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}

//-------------------------------------------------------------------

int test_q4(std::ostream &out, const std::string test_name) {
    std::string entity_name = "RandomGaussianMatrix";
    start_test_suite(out, test_name);

    std::vector<std::pair<int, int> > sizes = {std::make_pair(12, 4), std::make_pair(20, 5), std::make_pair(15, 4)};
    std::vector<int> res;

    for (auto it = sizes.begin(); it < sizes.end(); ++it) {
        int d = it->first;
        int l = it->second;
        Eigen::MatrixXd random_gaussian = RandomProjection::random_gaussian_matrix(d, l);
        res.push_back(test_eq(out, "rows", random_gaussian.rows(), d));
        res.push_back(test_eq(out, "cols", random_gaussian.cols(), l));
        res.push_back(test_eq_approx(out, "Mean", random_gaussian.mean(), 0., 0.2));
        res.push_back(test_eq_approx(out, "Stddev", 1.0 / (random_gaussian.size() - 1) * random_gaussian.squaredNorm(), 1.0 / l, 0.2));
    }

    return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}

//-------------------------------------------------------------------

int test_q5(std::ostream &out, const std::string test_name) {
    std::string entity_name = "RandomGaussianMatrix";
    start_test_suite(out, test_name);
    std::vector<int> res;

    // projection dim, filename, sampling type, col_class
    typedef std::tuple<int, const char*, std::string, int> TestCase;
    std::vector<TestCase> cases = {
        TestCase(20, "csv/mail_train.csv", "Gaussian", 0),
        TestCase(20, "csv/mail_train.csv", "Rademacher", 0)
    };

    for (auto it = cases.begin(); it != cases.end(); ++it) {
        int projection_dim = std::get<0>(*it);
        const char* dataset_file = std::get<1>(*it);
        std::string sampling = std::get<2>(*it);
        int col_class = std::get<3>(*it);

        // Puts train file in a Dataset object
	    Dataset train_dataset(dataset_file);

	    // Random projection
        clock_t t_random_projection = clock();
	    RandomProjection projection(train_dataset.get_dim() - 1, col_class, projection_dim, sampling);
        t_random_projection = clock() - t_random_projection;
        out << std::endl
            << "Execution time: "
            << (t_random_projection * 1000) / CLOCKS_PER_SEC
            << "ms\n\n";

	    // Trivial tests
        res.push_back(test_eq(out, "getOriginalDimension", projection.get_original_dimension(), train_dataset.get_dim() - 1));
	    res.push_back(test_eq(out, "getColClass", projection.get_col_class(), col_class));
        res.push_back(test_eq(out, "get_projectionsDim", projection.get_projection_dim(), projection_dim));
        res.push_back(test_eq(out, "getTypeSample", projection.get_type_sample(), sampling));

	    // Tests for projection matrix
        res.push_back(test_eq(out, "rows", projection.get_projection().rows(), train_dataset.get_dim()- 1));
        res.push_back(test_eq(out, "cols", projection.get_projection().cols(), projection_dim));
        res.push_back(test_eq_approx(out, "mean", projection.get_projection().mean(), 0., 0.15));
        res.push_back(test_eq_approx(
            out, "Stddev",
            1.0 / (projection.get_projection().size() - 1) * projection.get_projection().squaredNorm(),
            1.0/projection_dim,
            0.1));

	    // Uncomment this to try your ProjectionQuality bonus function!
        // projection.ProjectionQuality(&train_dataset);
    }

    return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}



int grading(std::ostream &out, const int test_case_number)
{
/**

Annotations used for the autograder.

[START-AUTOGRADER-ANNOTATION]
{
  "total" : 5,
  "names" : [
      "KnnClassification.cpp::test_q1",
      "KnnClassification.cpp::test_q2",
      "ConfusionMatrix.cpp::test_q3",
      "RandomProjection.cpp::test_q4",
      "RandomProjection.cpp::test_q5"
  ],
  "points" : [33, 33, 34, 0, 0]
}
[END-AUTOGRADER-ANNOTATION]
*/

    int const total_test_cases = 5;
    std::string const test_names[total_test_cases] = {"Test1", "Test2", "Test3", "Test4", "Test5"};
    int const points[total_test_cases] = {33, 33, 34, 0, 0};
    int (*test_functions[total_test_cases]) (std::ostream &, const std::string) = {
      test_q1, test_q2, test_q3, test_q4, test_q5
    };

    return run_grading(out, test_case_number, total_test_cases,
                       test_names, points,
                       test_functions);
}

} // End of namepsace tdgrading
