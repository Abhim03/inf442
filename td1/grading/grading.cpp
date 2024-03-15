#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstdarg>
#include <iterator>
#include <string>
#include <regex>
#include <numeric>
#include <cmath>
#include <fstream>
#include <vector>
#include <cassert>
#include <cstring>

#include "../gradinglib/gradinglib.hpp"
#include "../stats_functions.hpp"
#include <limits>

namespace tdgrading {

using namespace testlib;

double DELTA = 0.001;

int test_ex1(std::ostream &out, const std::string test_name) {
    std::string entity_name = "Ex 1: Basic statistic";
    start_test_suite(out, test_name);

    std::vector<int> res;
    double arr1[] = {1.0, 2.0, 3.0};
    double arr2[] = {1.0, 2.2, 3.45, -11.1, 2.0, 1.32};
    double arr3[] = {1.0};
    double arr4[] = {4.58, 3.6, 9.85, 7.37, 9.85, 1.7, 2.12, 3.68, 2.88, 6.11, 4.3};


    // Testing ComputeMean
    res.push_back(test_eq_approx(out, "ComputeMean", compute_mean(arr1, 3), 2.0, DELTA));
    res.push_back(test_eq_approx(out, "ComputeMean", compute_mean(arr2, 6), -0.18833, DELTA));
    res.push_back(test_eq_approx(out, "ComputeMean", compute_mean(arr3, 1), 1.0, DELTA));
    res.push_back(test_eq_approx(out, "ComputeMean", compute_mean(arr4, 11), 5.09454545, DELTA));

    // Testing ComputeVariance
    res.push_back(test_eq_approx(out, "ComputeVariance", compute_variance(arr1, 3), 0.6666, DELTA));
    res.push_back(test_eq_approx(out, "ComputeVariance", compute_variance(arr2, 6), 24.41368, DELTA));
    res.push_back(test_eq_approx(out, "ComputeVariance", compute_variance(arr3, 1), 0.0, DELTA));
    res.push_back(test_eq_approx(out, "ComputeVariance", compute_variance(arr4, 11), 7.4402975, DELTA));

    // Testing ComputeSampleVariance
    res.push_back(test_eq_approx(out, "ComputeSampleVariance", compute_sample_variance(arr1, 3), 1.0, DELTA));
    res.push_back(test_eq_approx(out, "ComputeSampleVariance", compute_sample_variance(arr2, 6), 29.29641667, DELTA));
    res.push_back(test_eq_approx(out, "ComputeSampleVariance", compute_sample_variance(arr4, 11), 8.18432727, DELTA));

    // Testing ComputeStandardDeviation
    res.push_back(test_eq_approx(out, "ComputeStandardDeviation", compute_standard_deviation(arr1, 3), 0.81649658, DELTA));
    res.push_back(test_eq_approx(out, "ComputeStandardDeviation", compute_standard_deviation(arr2, 6), 4.941, DELTA));
    res.push_back(test_eq_approx(out, "ComputeStandardDeviation", compute_standard_deviation(arr3, 1), 0.0, DELTA));
    res.push_back(test_eq_approx(out, "ComputeStandardDeviation", compute_standard_deviation(arr4, 11), 2.72769, DELTA));

    // Testing ComputeSampleStandardDeviation
    res.push_back(test_eq_approx(out, "ComputeSampleStandardDeviation", compute_sample_standard_deviation(arr1, 3), 1.0, DELTA));
    res.push_back(test_eq_approx(out, "ComputeSampleStandardDeviation", compute_sample_standard_deviation(arr2, 6), 5.4126, DELTA));
    res.push_back(test_eq_approx(out, "ComputeSampleStandardDeviation", compute_sample_standard_deviation(arr4, 11), 2.8608, DELTA));

    return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}

//--------------------------------------------------------

int test_ex2(std::ostream &out, const std::string test_name) {
    std::string entity_name = "Ex 2: Basic operations with matrices";

    start_test_suite(out, test_name);
    std::vector<int> res;

    double **mat1 = new double*[2];
    mat1[0] = new double[4]{1.0, 2.0, 3.0, 4.0};
    mat1[1] = new double[4]{5.0, 6.0, 7.0, 8.0};

    double **mat2 = new double*[4];;
    mat2[0] = new double[2]{1.1, 2.2};
    mat2[1] = new double[2]{1.3, 1.4};
    mat2[2] = new double[2]{1.4, 1.5};
    mat2[3] = new double[2]{2.0, 3.0};

    // Testing GetRow
    double row1e[4] = {1.0, 2.0, 3.0, 4.0};
    double row1s[4] = {0.0, 0.0, 0.0, 0.0};
    get_row(mat1, 4, 0, row1s);
    res.push_back(test_eq_array_ptr(out, "GetRow", row1s, row1e, 4));
    double row1ee[4] = {5.0, 6.0, 7.0, 8.0};
    get_row(mat1, 4, 1, row1s);
    res.push_back(test_eq_array_ptr(out, "GetRow", row1s, row1ee, 4));

    double row2e[2] = {1.3, 1.4};
    double row2s[2] = {0.0, 0.0};
    get_row(mat2, 2, 1, row2s);
    res.push_back(test_eq_array_ptr(out, "GetRow", row2s, row2e, 2));
    double row2ee[2] = {2.0, 3.0};
    get_row(mat2, 2, 3, row2s);
    res.push_back(test_eq_array_ptr(out, "GetRow", row2s, row2ee, 2));

    // Testing GetColumn
    double col1e[2] = {2.0, 6.0};
    double col1s[2] = {0.0, 0.0};
    get_column(mat1, 2, 1, col1s);
    res.push_back(test_eq_array_ptr(out, "GetColumn", col1s, col1e, 2));
    double col1ee[2] = {4.0, 8.0};
    get_column(mat1, 2, 3, col1s);
    res.push_back(test_eq_array_ptr(out, "GetColumn", col1s, col1ee, 2));

    double col2e[4] = {1.1, 1.3, 1.4, 2.0};
    double col2s[4] = {0.0, 0.0, 0.0, 0.0};
    get_column(mat2, 4, 0, col2s);
    res.push_back(test_eq_array_ptr(out, "GetColumn", col2s, col2e, 4));
    double col2ee[4] = {2.2, 1.4, 1.5, 3.0};
    get_column(mat2, 4, 1, col2s);
    res.push_back(test_eq_array_ptr(out, "GetColumn", col2s, col2ee, 4));

    for (int i = 0; i < 2; ++i)
        delete[] mat1[i];
    delete[] mat1;

    for (int i = 0; i < 4; ++i)
        delete[] mat2[i];
    delete[] mat2;

    return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}

//--------------------------------------------------------

int test_ex3(std::ostream &out, const std::string test_name) {
    std::string entity_name = "Ex 3: Covariance and correlation";

    start_test_suite(out, test_name);
    std::vector<int> res;

    double arr1[3] = {1.0, 3.0, 5.0};
    double arr2[3] = {5.0, 3.0, 1.0};
    double arr3[11] = {8.07, 4.8, 0.33, 3.44, 8.63, 2.4, 5.73, 0.47, 1.23, 9.01, 5.05};
    double arr4[11] = {7.08, 5.05, 3.8, 2.58, 5.57, 9.83, 9.03, 5.72, 6.08, 8.35, 2.81};

    // Test ComputeCovariance
    res.push_back(test_eq_approx(out, "ComputeCovariance", compute_covariance(arr1, arr1, 3), 2.666667, DELTA));
    res.push_back(test_eq_approx(out, "ComputeCovariance", compute_covariance(arr2, arr2, 3), 2.666667, DELTA));
    res.push_back(test_eq_approx(out, "ComputeCovariance", compute_covariance(arr1, arr2, 3), -2.666667, DELTA));
    res.push_back(test_eq_approx(out, "ComputeCovariance", compute_covariance(arr3, arr3, 11), 9.263, DELTA));
    res.push_back(test_eq_approx(out, "ComputeCovariance", compute_covariance(arr4, arr4, 11), 5.31113, DELTA));
    res.push_back(test_eq_approx(out, "ComputeCovariance", compute_covariance(arr3, arr4, 11), 1.816864, DELTA));

    // Test ComputeCorrelation
    res.push_back(test_eq_approx(out, "ComputeCorrelation", compute_correlation(arr1, arr1, 3), 1.0, DELTA));
    res.push_back(test_eq_approx(out, "ComputeCorrelation", compute_correlation(arr2, arr2, 3), 1.0, DELTA));
    res.push_back(test_eq_approx(out, "ComputeCorrelation", compute_correlation(arr1, arr2, 3), -1.0, DELTA));
    res.push_back(test_eq_approx(out, "ComputeCorrelation", compute_correlation(arr3, arr3, 11), 1.0, DELTA));
    res.push_back(test_eq_approx(out, "ComputeCorrelation", compute_correlation(arr4, arr4, 11), 1.0, DELTA));
    res.push_back(test_eq_approx(out, "ComputeCorrelation", compute_correlation(arr3, arr4, 11), 0.259, DELTA));

    return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}

//--------------------------------------------------------

int test_ex4(std::ostream &out, const std::string test_name) {
    std::string entity_name = "Ex 4: Covariance and correlation matrices";

    start_test_suite(out, test_name);
    std::vector<int> res;

    double** mat1 = new double*[3];
    mat1[0] = new double[3]{1.0, 2.0, 3.0};
    mat1[1] = new double[3]{4.0, 5.0, 6.0};
    mat1[2] = new double[3]{7.0, 8.0, 9.0};

    double** mat2 = new double*[5];
    mat2[0] = new double[5]{1.85, 0.41, 8.3 , 0.63, 2.69};
    mat2[1] = new double[5]{2.24, 2.4 , 6.28, 5.93, 5.9};
    mat2[2] = new double[5]{6.04, 3.71, 8.8 , 5.45, 0.68};
    mat2[3] = new double[5]{4.57, 9.01, 4.83, 7.64, 1.16};
    mat2[4] = new double[5]{4.2 , 0.83, 4.08, 5.89, 7.72};

    // Test ComputeCovarianceMatrix
    double correct1[3][3] = {{6., 6., 6.}, {6., 6., 6.}, {6., 6., 6.}};
    double** student_result1 = compute_covariance_matrix(mat1, 3, 3);
    for (int i = 0; i < 3; ++i)
            res.push_back(test_eq_array_ptr_approx(out, "ComputeCovarianceMatrix", student_result1[i], correct1[i], 3, DELTA));

    double correct2[5][5] = {
       {2.40092 ,  2.27276 , -0.05458 ,  2.09566 , -1.71642 },
       { 2.27276 ,  9.606256, -1.525036,  4.973604, -4.94838 },
       {-0.05458 , -1.525036,  3.442976, -2.715104, -2.94986 },
       { 2.09566 ,  4.973604, -2.715104,  5.573536,  0.40214 },
       {-1.71642 , -4.94838 , -2.94986 ,  0.40214 ,  7.5136  }
    };
    double** student_result2 = compute_covariance_matrix(mat2, 5, 5);
    for (int i = 0; i < 5; ++i)
            res.push_back(test_eq_array_ptr_approx(out, "ComputeCovarianceMatrix", student_result2[i], correct2[i], 5, DELTA));

    // Test ComputeCorrelationMatrix
    double correct_cor1[3][3] = {{1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}, {1.0, 1.0, 1.0}};
    double** student_result_cor1 = compute_correlation_matrix(mat1, 3, 3);
    for (int i = 0; i < 3; ++i)
            res.push_back(test_eq_array_ptr_approx(out, "ComputeCorrelationMatrix", student_result_cor1[i], correct_cor1[i], 3, DELTA));

    double correct_cor2[5][5] = {
        {1.0,  0.47324677, -0.01898356,  0.57288384, -0.40412032},
        {0.47324677,  1.0, -0.2651771 ,  0.67971792, -0.5824544},
        {-0.01898356, -0.2651771 ,  1.0, -0.61980356, -0.57997647},
        {0.57288384,  0.67971792, -0.61980356,  1.0,  0.06214237},
        {-0.40412032, -0.5824544 , -0.57997647,  0.06214237,  1.0}
    };
    double** student_result_cor2 = compute_correlation_matrix(mat2, 5, 5);
    for (int i = 0; i < 5; ++i)
            res.push_back(test_eq_array_ptr_approx(out, "ComputeCorrelationMatrix", student_result_cor2[i], correct_cor2[i], 5, DELTA));

    for (int i = 0; i < 3; ++i) {
        delete[] mat1[i];
        delete[] student_result1[i];
        delete[] student_result_cor1[i];
    }
    delete[] mat1;
    delete[] student_result1;
    delete[] student_result_cor1;

    for (int i = 0; i < 5; ++i) {
        delete[] mat2[i];
        delete[] student_result2[i];
        delete[] student_result_cor2[i];
    }
    delete[] mat2;
    delete[] student_result2;
    delete[] student_result_cor2;

    return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}

//--------------------------------------------------------

int grading(std::ostream &out, const int test_case_number)
{
	/**

	  Annotations used for the autograder.

	  [START-AUTOGRADER-ANNOTATION]
	  {
	  "total" : 4,
	  "names" : [
	  "stats_functions.cpp::test_ex1", 
	  "stats_functions.cpp::test_ex2", 
	  "stats_functions.cpp::test_ex3", 
	  "stats_functions.cpp::test_ex4"
	  ],
	  "points" : [25, 25, 25, 25]
	  }
	  [END-AUTOGRADER-ANNOTATION]
	  */

	int const total_test_cases = 4;
	std::string const test_names[total_test_cases] = {"Test_ex_one", "Test_ex_two", "Test_ex_three", "Test_ex_four"};
	int const points[total_test_cases] = {25, 25, 25, 25};
	int (*test_functions[total_test_cases]) (std::ostream &, const std::string) = {
		test_ex1, test_ex2, test_ex3, test_ex4
	};

	return run_grading(out, test_case_number, total_test_cases,
			test_names, points,
			test_functions);
}

} // End of namepsace tdgrading
