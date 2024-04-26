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
#include <random>
#include <limits>

#include <ANN/ANN.h>
#include <ANN/ANNperf.h>

#include "../gradinglib/gradinglib.hpp"
#include "Dataset.hpp"
#include "Regression.hpp"
#include "LinearRegression.hpp"
#include "KnnRegression.hpp"

namespace tdgrading {

using namespace testlib;
using namespace std;

const double deps = 0.001;
const std::string default_path = "./grading/tests/";    

std::vector<double> Boston_XTy = {590180.829446, 2840723.51761, 2156285.03048, 19039.9221446, 115274.278109, 1337892.85004, 14001845.827, 853578.915551, 
1673479.21376, 55239953.4765, 3854910.41925, 55264613.4912, 2525971.40407, 5019095.96678};

std::vector<std::vector<double>> Boston  = {
{ -1.43778, 0.0173683, 0.0334321, -1.43041, -6.76887, 2.05081, -0.022268, -0.531224, 0.608631, 0.000705713, -0.217417, -0.00288819, 0.19694, -0.180487 },
{ -8.42039, 0.230507, -0.349575, -0.76017, -8.7658, 2.78919, -0.133201, 6.51874, -0.601777, 0.0728535, -2.58498, -0.00013384, 0.516997, 0.542295 },
{ -6.31744, 0.0210511, -0.0165854, 1.55026, 21.8612, -0.830543, -0.00588946, -0.661229, -0.325042, 0.0210537, 0.365561, -0.00272097, 0.100265, 0.0304447 },
{ -0.118513, -0.00483676, -0.000193678, 0.00832503, 0.187571, 0.0088827, 0.000602191, 0.00711229, 0.0124081, -0.000420896, -0.0071473, -0.000112789, 0.000573135, 0.00592023 },
{ 0.827624, -0.000859078, -8.38266e-05, 0.00440634, 0.00704024, -0.000141677, 0.000798438, -0.0151188, 0.00326109, 0.000100344, -0.0122335, -0.000310831, 5.49852e-05, -0.00175903 },
{ 5.10504, 0.0192189, 0.00196949, -0.0123609, 0.0246179, -0.0104613, 0.00526083, -0.00151918, -0.0110603, 3.47998e-05, -0.00967672, 0.00150729, -0.0338487, 0.0413719 },
{ -43.2548, -0.294112, -0.13256, -0.123537, 2.35219, 83.0916, 7.41456, -4.16143, -0.300658, 0.0142501, 0.921566, -0.00116721, 1.52991, 0.032128 },
{ 11.4053, -0.0293058, 0.0270965, -0.0579315, 0.116035, -6.57167, -0.00894299, -0.0173815, 0.0070763, -2.69616e-05, 0.00235338, -0.00147648, -0.0229319, -0.0665445 },
{ -22.2919, 0.203405, -0.0151536, -0.172518, 1.22636, 8.58722, -0.394431, -0.00760759, 0.0428684, 0.037719, 0.405174, 0.01018, 0.0658516, 0.142617 },
{ 231.944, 0.0937454, 0.729198, 4.44159, -16.5349, 105.026, 0.493283, 0.14332, -0.0649218, 14.9925, 0.407642, -0.1126, -1.61171, -1.91816 },
{ 25.0653, -0.0276685, -0.024787, 0.0738822, -0.268992, -12.2666, -0.131407, 0.00887946, 0.00542887, 0.154286, 0.000390526, 0.00258834, -0.0357258, -0.095727 },
{ 386.959, -0.18408, -0.000642746, -0.275418, -2.12595, -156.095, 10.2512, -0.00563243, -1.70581, 1.94142, -0.0540253, 1.29631, 0.639737, -0.128924 },
{ 31.0254, 0.126716, 0.0250646, 0.102456, 0.109059, 0.278758, -2.32401, 0.07453, -0.267462, 0.126782, -0.00780663, -0.180629, 0.00645831, -0.317104 },
{ 30.1835, -0.194652, 0.0440677, 0.0521448, 1.88823, -14.9475, 4.76119, 0.00262339, -1.30091, 0.46023, -0.0155731, -0.811248, -0.00218155, -0.531514 }
};

std::vector<std::vector<double>> RedWine  = {
{ -660.373, -0.296525, 1.39186, -0.238212, -4.45294, 0.00673006, -0.0076202, 685.417, -5.49347, -0.600104, 0.474934, 0.0309372 },
{ -33.884, -0.0138812, -0.560518, -0.0139507, 0.441621, -0.00172112, 0.000806976, 35.2681, -0.194332, -0.163726, 0.0498344, -0.0437634 },
{ -8.54648, 0.0521218, -0.448383, -0.00434468, 0.731694, -0.00254118, 0.00153703, 8.67994, -0.138677, -0.0489741, 0.0420753, -0.0095979 },
{ -845.654, -0.659516, -0.82507, -0.321214, -2.02094, 0.00924564, 0.00325395, 862.06, -3.59827, -0.978333, 0.679856, 0.0318215 },
{ -9.29178, -0.0235173, 0.0498226, 0.103192, -0.00385509, 0.000692777, -0.000436164, 10.0651, -0.156542, 0.10312, 0.00019447, -0.00772977 },
{ -395.822, 0.78131, -4.26826, -7.87801, 0.387687, 15.2285, 0.234759, 353.028, 10.4742, -0.488434, 0.407861, 0.718078 },
{ 193.745, -8.25316, 18.6702, 44.4543, 1.27293, -89.4462, 2.19013, 89.5289, -54.4989, 12.6418, -0.701476, -6.00554 },
{ 0.97798, 0.000927994, 0.00102002, 0.000313822, 0.000421568, 0.00258029, 4.11713e-06, 1.11918e-07, 0.00473951, 0.00100506, -0.000677029, -1.4988e-05 },
{ -56.8251, -0.0953355, -0.072042, -0.0642669, -0.0225549, -0.514395, 0.00156576, -0.000873256, 60.7506, -0.105931, 0.0621358, -0.00326131 },
{ -50.028, -0.0419544, -0.244514, -0.091431, -0.0247046, 1.36506, -0.000294139, 0.000816029, 51.8983, -0.426744, 0.0527067, 0.0438874 },
{ 555.507, 0.540446, 1.21139, 1.27856, 0.279432, 0.0419017, 0.00399785, -0.000737019, -569.031, 4.07431, 0.857895, 0.306936 },
{ 15.4472, 0.0335424, -1.01358, -0.277886, 0.0124616, -1.58686, 0.00670627, -0.0060119, -12.0024, -0.203751, 0.680616, 0.292444 }
};

std::vector<std::vector<double>> WhiteWine  = {
{ -710.944, -0.484234, 0.468924, -0.257376, -4.30806, 0.00168227, -0.00234046, 727.343, -3.53076, -1.05186, 0.799428, 0.0543195 },
{ -6.58946, -0.0185883, -0.127927, 0.000752747, 0.287754, -0.00117442, 0.000432157, 7.28862, -0.121039, -0.0224521, 0.0313838, -0.0279967 },
{ -12.9908, 0.0309728, -0.220118, -0.000974285, 0.493541, 0.000585825, -7.61557e-05, 13.3544, -0.130398, 0.0520595, 0.0275065, -0.000760758 },
{ -2475.04, -1.91299, 0.145751, -0.109636, -13.1609, 0.020899, -0.00733121, 2511.52, -8.53586, -4.19374, 2.48714, 0.304874 },
{ -7.93796, -0.00869786, 0.0151345, 0.0150861, -0.00357495, 9.9206e-05, 2.16721e-06, 8.20058, -0.0370071, -0.0106866, 0.00243882, -0.000302472 },
{ 3616.31, 1.00007, -18.1876, 5.27259, 1.67153, 29.2106, 0.242093, -3653.2, 12.3275, -5.87694, -4.2075, 1.37346 },
{ -13970.1, -8.45229, 40.6567, -4.16388, -3.56208, 3.87653, 1.47069, 14238.1, -45.5499, 31.1846, 10.3219, -0.0507728 },
{ 0.986425, 0.000764929, 0.000199684, 0.000212632, 0.000355364, 0.00427165, -6.46281e-06, 4.1463e-06, 0.00329204, 0.00154683, -0.00106636, -9.69113e-05 },
{ -124.084, -0.144269, -0.128839, -0.0806677, -0.0469253, -0.74896, 0.000847317, -0.000515369, 127.905, -0.0732896, 0.142751, 0.0187976 },
{ -67.0824, -0.0485394, -0.0269903, 0.0363712, -0.026037, -0.244256, -0.000456196, 0.000398476, 67.8729, -0.0827698, 0.0691344, 0.0153725 },
{ 705.784, 0.562836, 0.575604, 0.293197, 0.23559, 0.850454, -0.00498302, 0.00201229, -713.881, 2.45966, 1.05478, 0.0339177 },
{ 238.253, 0.141813, -1.90407, -0.0300696, 0.107086, -0.391123, 0.00603171, -3.67043e-05, -240.576, 1.20104, 0.869698, 0.125772 }
};

double rel_error(double a, double b) {
    return fabs(a - b) / fabs(a);
}

template <typename T, typename... Arguments>
bool test_rel_error(std::ostream &out,
                      const std::string &function_name,
                      T result,
                      T expected,
                      T delta,
                      const Arguments&... args) 
{
    bool success = (rel_error(result, expected) <= delta);
    
    out << (success ? "[SUCCESS] " : "[FAILURE] ");

    print_tested_function(out, function_name, args...);

    out << ": got " << result
        << " expected " << expected << "  The relative error should be in [-" << delta << "," << delta << "]";
    out << std::endl;

    return success;
}

void test_construction_X_y(const std::string fname,
                           std::vector<double>&  B, // true values of beta
                           const bool verbose, const double eps,
                           std::ostream& out, std::vector<int>& res) 
{
    cout << endl << endl;
    cout << "Constructing X and y for Linear Regression for the dataset " << fname <<  endl;
    Dataset train_dataset(fname.c_str());
    
    train_dataset.show(false);  // only dimensions and samples
    
    for (int col_regr = 0; col_regr < (int)B.size(); ++col_regr) {

        cout << endl << endl;
        std::cout << "Linear Regression over column " << col_regr << std::endl;
        LinearRegression tester(&train_dataset, col_regr);

        double n1;
        Eigen::MatrixXd X = tester.construct_matrix();
        Eigen::VectorXd y = tester.construct_y();
        if ((X.rows() == 0) || (X.cols() == 0) || (y.size() == 0)) n1 = 0.0;
        else n1 = (X.transpose() * y).norm();

        res.push_back(test_eq(out, "Construct X, column of 1 ", X(0, 0), 1.));
        res.push_back(test_eq(out, "Construct X, # rows ", X.rows(), train_dataset.get_nbr_samples()));
        res.push_back(test_eq(out, "Construct X, # columns ", X.cols(), train_dataset.get_instance(0).size()));
        res.push_back(test_eq(out, "Construct y, # rows ", y.rows(), train_dataset.get_nbr_samples()));
        res.push_back(test_eq_approx(out, "Test X^T*y) ", n1, B[col_regr], deps));
    }
    cout << endl;
    return;
}

int Ex1(std::ostream &out, const std::string test_name) 
{
    std::string entity_name = "Linear Regression :: contruct X and y";
    start_test_suite(out, test_name);
    std::vector<int> res; 

    test_construction_X_y("./csv/train_boston_housing.csv", Boston_XTy, true, deps, out, res);

    return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}

void test_Regression(const std::string fname,
                     std::vector<std::vector<double>>&  B, // true values of beta
                     const bool verbose, const double eps,
                     std::ostream& out, std::vector<int>& res) 
{
    cout << endl << endl;
    cout << "Linear Regression on the dataset " << fname <<  endl;
    Dataset train_dataset(fname.c_str());
    
    train_dataset.show(false);  // only dimensions and samples
    
    for (int col_regr = 0; col_regr < (int)B.size(); ++col_regr) {

        cout << endl  << endl;
       // std::cout << "Linear Regression over column " << col_regr << std::endl;
        LinearRegression tester(&train_dataset, col_regr);

        Eigen::VectorXd beta = 
        Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(B[col_regr].data(), B[col_regr].size());

        double RE = (beta - *tester.get_coefficients()).norm() / beta.norm();
        res.push_back(test_eq_approx(out, "Linear Regression", RE, 0.0, deps));
    }
    cout << endl;
    return ;
}

int Ex2(std::ostream &out, const std::string test_name) 
{
    std::string entity_name = "Linear Regression :: compute beta";
    start_test_suite(out, test_name);
    std::vector<int> res; 

    test_Regression("./csv/train_boston_housing.csv", Boston, true, deps, out, res);
    test_Regression("./csv/train_winequality-red.csv", RedWine, true, deps, out, res);
    test_Regression("./csv/train_winequality-white.csv", WhiteWine, true, deps, out, res);

    return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}

struct rData {
    int col_regr;
    int idx ;                       // index of the sample
    double tv;                      // true value;
    double ev;                      // estimated value by the regressor;
    double at_ess, at_rss, at_tss;  // errors for the training set
    double ar_ess, ar_rss, ar_tss;  // errors for the Regression set
};

void test_estimate(const std::string fname1, const std::string fname2,
                     std::vector<std::vector<double>>&  B, // true values of beta
                     const rData& data,
                     const bool verbose, const double eps,
                     std::ostream& out, std::vector<int>& res) 
{
    cout << endl << endl;
    cout << "Linear Regression and Estimation with files " << fname1 << " " << fname2 << endl;
    Dataset train_dataset(fname1.c_str());
    Dataset regr_dataset(fname2.c_str());
    
    train_dataset.show(false);  // only dimensions and samples
    // Checks if train and test are same format
    assert((train_dataset.get_dim() == regr_dataset.get_dim()));  // otherwise doesn't make sense

    cout << endl  << endl;
    std::cout << "Linear Regression over column " << data.col_regr << std::endl;
    LinearRegression tester(&train_dataset, data.col_regr);

    Eigen::VectorXd beta = 
    Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(B[data.col_regr].data(), B[data.col_regr].size());
    double RE = (beta - *tester.get_coefficients()).norm() / beta.norm();
    bool b1 = test_eq_approx(out, "Linear Regression", RE, 0.0, deps);

    std::vector<double> sample = regr_dataset.get_instance(data.idx);
        
    Eigen::VectorXd tX(regr_dataset.get_dim() - 1);
    int j = 0;
    for (int i = 0; i < regr_dataset.get_dim(); i++) {
        if (i != data.col_regr) {
            tX(j) = sample[i];
            j++;
        }
    }

    double cv = tester.estimate(tX);  // computed value
    cout << "Regressor should compute " << data.ev << " while the real value is " << data.tv <<  endl;
    bool b2 = test_eq_approx(out, "Linear Regression estimate", cv, data.ev, deps); 

    res.push_back( b1 && b2);
    return ;
}

int Ex3(std::ostream &out, const std::string test_name) 
{
    std::string entity_name = "Linear Regression :: estimate";
    start_test_suite(out, test_name);
    std::vector<int> res; 

    struct rData Boston_data = { 13, 0, 5, 4.24626, 25600.5, 9208.56, 34809.1, 4321.93, 3364.67, 2685.42 };
    struct rData Red_data = {11, 0, 6, 5.4228, 134.16, 242.275, 376.435, 262.65, 446.862, 665.017};
    struct rData White_data = {11, 0, 5, 5.88042, 499.758, 1219.23, 1718.99, 1004.18, 1661.4, 2121.33};

    test_estimate("./csv/train_boston_housing.csv", "./csv/regr_boston_housing.csv",
                    Boston, Boston_data,
                    true, deps, out, res);
    test_estimate("./csv/train_winequality-red.csv", "./csv/regr_winequality-red.csv",
                     RedWine, Red_data,
                     true, deps, out, res);
    test_estimate("./csv/train_winequality-white.csv", "./csv/regr_winequality-white.csv",
                     WhiteWine, White_data,
                     true, deps, out, res);

    return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}

void test_LR_errors(const std::string fname1, const std::string fname2,
                     std::vector<std::vector<double>>&  B, // true values of beta
                     const rData& data,
                     const bool verbose, const double eps,
                     std::ostream& out, std::vector<int>& res) 
{
    cout << endl << endl;
    cout << "Linear Regression and Estimation with files " << fname1 << " " << fname2 << endl;
    Dataset train_dataset(fname1.c_str());
    Dataset regr_dataset(fname2.c_str());
    
    train_dataset.show(false);  // only dimensions and samples
    // Checks if train and test are same format
    assert((train_dataset.get_dim() == regr_dataset.get_dim()));  // otherwise doesn't make sense

    cout << endl  << endl;
    std::cout << "Linear Regression over column " << data.col_regr << std::endl;
    LinearRegression tester(&train_dataset, data.col_regr);

    Eigen::VectorXd beta = 
    Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(B[data.col_regr].data(), B[data.col_regr].size());

    double RE = (beta - *tester.get_coefficients()).norm() / beta.norm();
    bool b1 = test_eq_approx(out, "Linear Regression", RE, 0.0, deps);

    std::vector<double> sample = regr_dataset.get_instance(data.idx);
        
    Eigen::VectorXd tX(regr_dataset.get_dim() - 1);
    int j = 0;
    for (int i = 0; i < regr_dataset.get_dim(); i++) {
        if (i != data.col_regr) {
            tX(j) = sample[i];
            j++;
        }
    }

    double cv = tester.estimate(tX);  // computed value
    cout << "Regressor should compute " << data.ev << " while the real value is " << data.tv <<  endl;
    bool b2 = test_eq_approx(out, "Linear Regression estimate", cv, data.ev, deps);

    // Train set
    double ct_ess, ct_rss, ct_tss;
    tester.sum_of_squares(&train_dataset, ct_ess, ct_rss, ct_tss);
    bool tb1 = test_rel_error(out, "Linear Regression SOS", data.at_ess, ct_ess, deps);
    bool tb2 = test_rel_error(out, "Linear Regression SOS", data.at_rss, ct_rss, deps);
    bool tb3 = test_rel_error(out, "Linear Regression SOS", data.at_tss, ct_tss, deps);
    bool tb  = tb1 && tb2 && tb3;

    // Reg set
    double cr_ess, cr_rss, cr_tss;
    tester.sum_of_squares(&regr_dataset, cr_ess, cr_rss, cr_tss);
    bool rb1 = test_rel_error(out, "Linear Regression SOS", data.ar_ess, cr_ess, deps);
    bool rb2 = test_rel_error(out, "Linear Regression SOS", data.ar_rss, cr_rss, deps);
    bool rb3 = test_rel_error(out, "Linear Regression SOS", data.ar_tss, cr_tss, deps);
    bool rb  = rb1 && rb2 && rb3;

    res.push_back(b1 && b2 && tb && rb);

    return ;
}

int Ex4(std::ostream &out, const std::string test_name) 
{
    std::string entity_name = "Linear Regression errors in datasets";
    start_test_suite(out, test_name);
    std::vector<int> res; 

    struct rData Boston_data = { 13, 0, 5, 4.24626, 25600.5, 9208.56, 34809.1, 4321.93, 3364.67, 2685.42 };
    struct rData Red_data = {11, 0, 6, 5.4228, 134.16, 242.275, 376.435, 262.65, 446.862, 665.017};
    struct rData White_data = {11, 0, 5, 5.88042, 499.758, 1219.23, 1718.99, 1004.18, 1661.4, 2121.33};

    test_LR_errors("./csv/train_boston_housing.csv", "./csv/regr_boston_housing.csv",
                    Boston, Boston_data,
                    true, deps, out, res);
    test_LR_errors("./csv/train_winequality-red.csv", "./csv/regr_winequality-red.csv",
                     RedWine, Red_data,
                     true, deps, out, res);
    test_LR_errors("./csv/train_winequality-white.csv", "./csv/regr_winequality-white.csv",
                     WhiteWine, White_data,
                     true, deps, out, res);

    return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}

void test_Knn_ctor(const std::string fname,
                    int col_class, 
                    int k,
                     const bool verbose, const double eps,
                     std::ostream& out, std::vector<int>& res) 
{
    cout << endl << endl;
    cout << "Knn Regression on the dataset " << fname <<  endl;
    Dataset train_dataset(fname.c_str());
    
    train_dataset.show(false);  // only dimensions and samples
    
    // Regression
    std::cout << "Computing k-NN Regression (k=" << k << ", Regression over column " << col_class << ")..." << std::endl;
    KnnRegression knn_regr(k, &train_dataset, col_class);

    // Tests

    bool b1 = test_eq(out, "Knn Regression col", knn_regr.get_col_regr(), col_class);
    bool b2 = test_eq(out, "Knn  Regression k", knn_regr.get_k(), k);
    
    res.push_back(b1 && b2);

    std::cout << std::endl
              << "Statistics for the ANN kd-tree" << endl;
    ANNkdStats stats;
    knn_regr.get_kdTree()->getStats(stats);
    std::cout << stats.dim << " : dimension of space (e.g. 11 for train_winequality-red)" << std::endl;
    std::cout << stats.n_pts << " : no. of points (e.g. 598 for train_winequality-red)" << std::endl;
    std::cout << stats.bkt_size << " : bucket size" << std::endl;
    std::cout << stats.n_lf << " : no. of leaves (including trivial)" << std::endl;
    std::cout << stats.n_tl << " : no. of trivial leaves (no points)" << std::endl;
    std::cout << stats.n_spl << " : no. of splitting nodes" << std::endl;
    std::cout << stats.n_shr << " : no. of shrinking nodes (for bd-trees)" << std::endl;
    std::cout << stats.depth << " : depth of tree" << std::endl;
    std::cout << stats.sum_ar << " : sum of leaf aspect ratios" << std::endl;
    std::cout << stats.avg_ar << " : average leaf aspect ratio" << std::endl;

    annClose();
    return ;
}

int Ex5(std::ostream &out, const std::string test_name) 
{
    std::string entity_name = "Knn Regression constructor/destructor";
    start_test_suite(out, test_name);
    std::vector<int> res; 

    test_Knn_ctor("./csv/train_boston_housing.csv", 13, 3, true, deps, out, res);
    test_Knn_ctor("./csv/train_winequality-red.csv", 10, 4, true, deps, out, res);
    test_Knn_ctor("./csv/train_winequality-white.csv", 11, 5, true, deps, out, res);

    return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}

void test_Knn_Regression(const std::string fname1,
                        const std::string fname2,
                        int col_regr, 
                        int k,
                        double amse,
                        const bool verbose, const double eps,
                        std::ostream& out, std::vector<int>& res) 
{
    cout << endl << endl;
    cout << "Knn Regression on the dataset " << fname1 <<  " (and " << fname2 << ")" << endl;
    Dataset train_dataset(fname1.c_str());
    Dataset regr_dataset(fname2.c_str());

    train_dataset.show(false);  // only dimensions and samples
    assert(train_dataset.get_dim() == regr_dataset.get_dim());  // otherwise doesn't make sense
    
    // Regression
    std::cout << "Computing k-NN Regression (k=" << k << ", Regression over column " << col_regr << ")..." << std::endl;
    KnnRegression knn_regr(k, &train_dataset, col_regr);

    double cmse = 0;
    for (int i = 0; i < regr_dataset.get_nbr_samples(); i++) {
        std::vector<double> sample = regr_dataset.get_instance(i);
        // extract column for Regression
        Eigen::VectorXd query(regr_dataset.get_dim() - 1);
        for (int j = 0, j2 = 0; j < train_dataset.get_dim() - 1 && j2 < train_dataset.get_dim(); j++, j2++) {
            if (j == col_regr && j2 == col_regr) {
                j--;
                continue;
            }
            query[j] = sample[j2];
        }
        double estim = knn_regr.estimate(query);
        cmse += (estim - sample[col_regr]) * (estim - sample[col_regr]) / regr_dataset.get_nbr_samples();
    }

    annClose();
    std::cout << "Mean Square(d) Error (MSE) over test set: " << cmse << std::endl;

    // Tests
    res.push_back(test_rel_error(out, "Mean Square(d) Error (MSE)", amse, cmse, deps));


    // std::cout << std::endl
    //           << "Statistics for the ANN kd-tree" << endl;
    // ANNkdStats stats;
    // knn_regr.get_kdTree()->getStats(stats);
    // std::cout << stats.dim << " : dimension of space (e.g. 11 for train_winequality-red)" << std::endl;
    // std::cout << stats.n_pts << " : no. of points (e.g. 598 for train_winequality-red)" << std::endl;
    // std::cout << stats.bkt_size << " : bucket size" << std::endl;
    // std::cout << stats.n_lf << " : no. of leaves (including trivial)" << std::endl;
    // std::cout << stats.n_tl << " : no. of trivial leaves (no points)" << std::endl;
    // std::cout << stats.n_spl << " : no. of splitting nodes" << std::endl;
    // std::cout << stats.n_shr << " : no. of shrinking nodes (for bd-trees)" << std::endl;
    // std::cout << stats.depth << " : depth of tree" << std::endl;
    // std::cout << stats.sum_ar << " : sum of leaf aspect ratios" << std::endl;
    // std::cout << stats.avg_ar << " : average leaf aspect ratio" << std::endl;

    annClose();
    return ;
}  

int Ex6(std::ostream &out, const std::string test_name) 
{
    std::string entity_name = "k-nearest neighbors and Regression";
    start_test_suite(out, test_name);
    std::vector<int> res; 

    test_Knn_Regression("./csv/train_boston_housing.csv", "./csv/regr_boston_housing.csv", 
                        13, 3, 35.0232, true, deps, out, res);
    test_Knn_Regression("./csv/train_boston_housing.csv", "./csv/regr_boston_housing.csv", 
                        13, 5, 31.2295, true, deps, out, res);
    test_Knn_Regression("./csv/train_boston_housing.csv", "./csv/regr_boston_housing.csv", 
                        13, 10, 28.4425, true, deps, out, res);                        
    test_Knn_Regression("./csv/train_winequality-red.csv", "./csv/regr_winequality-red.csv", 
                        11, 3, 0.708514, true, deps, out, res);
    test_Knn_Regression("./csv/train_winequality-red.csv", "./csv/regr_winequality-red.csv", 
                        11, 5, 0.647433, true, deps, out, res);
    test_Knn_Regression("./csv/train_winequality-red.csv", "./csv/regr_winequality-red.csv", 
                        11, 10, 0.641149, true, deps, out, res);
    test_Knn_Regression("./csv/train_winequality-white.csv", "./csv/regr_winequality-white.csv", 
                        11, 3, 0.963231, true, deps, out, res);
    test_Knn_Regression("./csv/train_winequality-white.csv", "./csv/regr_winequality-white.csv", 
                        11, 5, 0.84421, true, deps, out, res);
    test_Knn_Regression("./csv/train_winequality-white.csv", "./csv/regr_winequality-white.csv", 
                        11, 10, 0.734569, true, deps, out, res);
   
    return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}

int grading(std::ostream &out, const int test_case_number)
{
/**

Annotations used for the autograder.

[START-AUTOGRADER-ANNOTATION]
{
  "total" : 6,
  "names" : [
        "LinearRegression.cpp::Ex1", 
        "LinearRegression.cpp::Ex2",
        "LinearRegression.cpp::Ex3",
        "KnnRegression.cpp::Ex4",
        "KnnRegression.cpp::Ex5",
        "KnnRegression.cpp::Ex6"
        ],
  "points" : [10,10,10,10,10,10]
}
[END-AUTOGRADER-ANNOTATION]
*/

    int const total_test_cases = 6;
    std::string const test_names[total_test_cases] = {"Ex1", "Ex2", "Ex3", "Ex4", "Ex5", "Ex6"};
    int const points[total_test_cases] = {10, 10, 10, 10, 10, 10};
    int (*test_functions[total_test_cases]) (std::ostream &, const std::string) = {
      Ex1, Ex2, Ex3, Ex4, Ex5, Ex6
    };

    return run_grading(out, test_case_number, total_test_cases,
                       test_names, points,
                       test_functions);
}

} // End of namepsace tdgrading
