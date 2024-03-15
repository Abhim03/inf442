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

#include "../gradinglib/gradinglib.hpp"
#include "../retrieval.hpp"


namespace tdgrading {

using namespace testlib;
using namespace std;

const double deps = 0.001;
const std::string default_path = "./grading/tests/";    


void test_dist_from_file(const std::string fname, const bool verbose, const double eps,
                        std::ostream& out, std::vector<int>& res) 
{

    // Test the distance function using data from a file
    // The format of the file is as follows:
    // N (= #tests)
    // dim
    // p_1,..., p_dim, q_1,..., q_dim, dist(p, q)  (repeated N times)
    std::cout << std::endl << std::endl << std::endl;
    std::cout << "Testing the function dist()...\t\t\n";
    std::cout << "  Using data from the file " << fname << std::endl;

    int nb_errors = 0;
    std::ifstream in(fname, std::ios_base::in);
    int N;  // number tests
    int dim;
    in >> N;
    in >> dim;
    std::cout << "  " << N << " pairs of points in dimension " << dim << std::endl;

    point p = new double[dim];
    point q = new double[dim];

    double adist = 0.0;
    double cdist = 0.0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < dim; j++) in >> p[j];
        for (int j = 0; j < dim; j++) in >> q[j];
        in >> adist;
        cdist = dist(p, q, dim);

        if (verbose) {
            std::cout << std::endl
                      << "(" << i  << ")  Distance of the points " << std::endl;
            print_point(p, dim);
            print_point(q, dim);
        }

        res.push_back(test_eq_approx(out, "dist", cdist, adist, eps));     
    }
    delete[] p;
    delete[] q;
    return;
}

int test_Ex_1(std::ostream &out, const std::string test_name) {
    std::string entity_name = "kd-tree";
    start_test_suite(out, test_name);

    std::vector<int> res; 
    test_dist_from_file(default_path + "dist_4.dat", false, deps, out, res);    
    test_dist_from_file(default_path + "dist_10.dat", false, deps, out, res); 

    return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}



void test_linear_scan_from_file(const std::string fname, bool verbose, const double eps,
                        std::ostream& out, std::vector<int>& res) 
{
    std::cout << std::endl << std::endl << std::endl;
    std::cout << "Testing the function linear_scan()...\t\t\n";
    std::cout << "  Using data from the file " << fname << std::endl;

    std::ifstream in(fname, std::ios_base::in);

    int N;    // number of points
    int dim;  // dimension
    int T;    // number of tests
    in >> N;
    in >> dim;
    std::cout << "  " << N << " #points in dimension " << dim << std::endl;
    in >> T;
    std::cout << "  There are " << T << " tests \n";

    point P[10000];
    for (int i = 0; i < N; i++) {
        P[i] = new double[dim];
        for (int j = 0; j < dim; j++) {
            in >> P[i][j];
        }
    }

    point q = new double[dim];
    int nb_errors = 0;
    for (int t = 0; t < T; t++) {
        for (int j = 0; j < dim; j++) {
            in >> q[j];
        }
        int aidx;
        double adist;
        in >> aidx >> adist;

        int cidx = linear_scan(q, dim, P, N);

        if (verbose) {
            std::cout << std::endl
                      << "(" << t << ")  For the query point " << std::endl;
            print_point(q, dim);
            std::cout << "the NN has index : (actual, computed) = ( "
                      << aidx << ", " << cidx << ")" << endl;
        }
        res.push_back(test_eq(out, "linear_scan", cidx, aidx));     
    }

    delete[] q;
    return ;
}

int test_Ex_2(std::ostream &out, const std::string test_name) {
    std::string entity_name = "kd-tree";
    start_test_suite(out, test_name);

    std::vector<int> res;

    test_linear_scan_from_file(default_path + "t_ls.dat", false, deps, out, res);
    test_linear_scan_from_file(default_path + "ls-1000-100-100.dat", false, deps, out, res);
    // test_linear_scan_from_file(default_path + "ls-1000-300-100.dat", true, deps, out, res);

    return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}



void test_median_from_file(const std::string fname, bool verbose, const double eps,
                        std::ostream& out, std::vector<int>& res) 
{
    std::cout << std::endl << std::endl << std::endl;
    std::cout << "Testing the function compute_median()...\t\t\n";
    std::cout << "  Using data from the file " << fname << std::endl;

    int nb_errors = 0;
    std::ifstream in(fname, std::ios_base::in);
    int N;    // number of points
    int dim;  // dimension
    int T;    // number of tests
    in >> N;
    in >> dim;
    std::cout << "  " << N << " #points in dimension " << dim << std::endl;
    in >> T;
    std::cout << "  There are " << T << " tests for computing the median \n";

    point P[20000];

    for (int i = 0; i < N; i++) {
        P[i] = new double[dim];
        for (int j = 0; j < dim; j++) {
            in >> P[i][j];
        }
        //for (int k=0; k < dim ; k++) { std::cout << P[i][k] << " ";  } ; std::cout << " \n";
    }

    for (int i = 0; i < T; i++) {
        int s, e, c;
        double amedian, cmedian;  // actual and computed median;
        in >> s >> e >> c >> amedian;
        cmedian = compute_median(P, s, e, c);
       
        if (verbose) {
            std::cout << std::endl
                      << "(" << i + 1 << ")  For the range ["
                      << s << " .. " << e << ") and coord " << c
                      << " the median is (computed,  actual) = ("
                      << cmedian << ", " << amedian << ")" << std::endl;
        }
        res.push_back(test_eq_approx(out, "compute_median", cmedian, amedian, eps));     

    }
    for (int i = 0; i < N; i++) {
        delete[] P[i];
    }
    return;
}

int test_Ex_3(std::ostream &out, const std::string test_name) {
    std::string entity_name = "kd-tree";
    start_test_suite(out, test_name);

    std::vector<int> res;
    test_median_from_file(default_path + "t_median.dat", false, deps, out, res);
    test_median_from_file(default_path + "median-01.dat", false, deps, out, res);

    return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}


void test_partition_from_file(const std::string fname, bool verbose, const double eps,
                        std::ostream& out, std::vector<int>& res)
{
    std::cout << std::endl << std::endl << std::endl;
    std::cout << "Testing the function partition()...\t\t\n";
    std::cout << "  Using data from the file " << fname << std::endl;

    std::ifstream in(fname, std::ios_base::in);
    int N;    // number of points
    int dim;  // dimension
    int T;    // number of tests
    in >> N;
    in >> dim;
    std::cout << "  " << N << " #points in dimension " << dim << std::endl;
    in >> T;
    std::cout << "  There are " << T << " tests for computing the median and partition \n";

    point P[10000];
    point Q[10000];  // A copy of the P

    int nb_errors_m = 0;  // number of errors in median computations
    int nb_errors_p_left = 0;  // number of errors in partition computations (on the left of the median)
    int nb_errors_p_right = 0;  // number of errors in partition computations (on the right of the median)


    for (int i = 0; i < N; i++) {
        P[i] = new double[dim];
        Q[i] = new double[dim];
        for (int j = 0; j < dim; j++) {
            in >> P[i][j];
            Q[i][j] = P[i][j];
        }
    }

    for (int t = 0; t < T; t++) {
        //for (int k=0; k < dim ; k++) { std::cout << P[i][k] << " ";  } ; std::cout << " \n";
        int s, e, c;
        int aidx, cidx; // actual and computed index (of the median) in P
        double amedian, cmedian;  // actual and computed median;
        in >> s >> e >> c >> aidx >> amedian;
        cmedian = compute_median(Q, s, e, c);
        if (fabs(amedian - cmedian) > eps) {
            ++nb_errors_m;
        }
        cidx = partition(Q, s, e, c);
        // Check if to the left of median the elements are <=
        for (int j = s; j <= cidx; j++) {
            if (Q[j][c] > cmedian) {
                nb_errors_p_left++;
            }
        }
        // Check if to the right of median the elements are >
        for (int j = cidx + 1; j < e; j++) {
            if (Q[j][c] < cmedian) {
                nb_errors_p_right++;
            }
        }

        if (verbose) {
            std::cout << std::endl
                      << "(" << t << ")  For the range ["
                      << s << " .. " << e << ") and coord " << c
                      << " the median is (computed,  actual) = ("
                      << cmedian << ", " << amedian << ")" << std::endl;
        }

        bool b1 = test_eq_approx(out, "median (in partition)", cmedian, amedian, eps);
        bool b2 = (cidx < 0 ? 0 : test_eq_approx(out, "does the median and its index agree", Q[cidx][c], cmedian, eps));
        bool b3 = test_eq(out, "partition (are the elements on the left smaller)", (nb_errors_p_left == 0), true);
        bool b4 = test_eq(out, "partition (are the elements on the right greater)", (nb_errors_p_right == 0), true);
       

        res.push_back(b1 && b2 && b3 && b4);

        for (int i = s; i < e; i++) {
            for (int j = 0; j < dim; j++) {
                Q[i][j] = P[i][j];
            }
        }
    }
    for (int i = 0; i < N; i++) {
        delete[] P[i];
        delete[] Q[i];
    }

    return;
}

int test_Ex_4(std::ostream &out, const std::string test_name) {
    std::string entity_name = "kd-tree";
    start_test_suite(out, test_name);

    std::vector<int> res;
    test_partition_from_file(default_path + "t_partition.dat", false, deps, out, res);
    test_partition_from_file(default_path + "partition-01.dat", false, deps, out, res);
  
    return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}


void test_create_node(int T, std::ostream& out, std::vector<int>& res) 
{
    std::cout << std::endl << std::endl << std::endl;
    std::cout << "Testing the function create_node() on " 
    << T << " tests ...\t\t\n";

    int nb_errors_1 = 0;
    int nb_errors_2 = 0;
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<int> Ndis(0, 1000);

    for (int i = 0; i < T; i++) {
        int idx = Ndis(generator);

        node* n = create_node(idx);
        // Test if we created the node correctly
        if ((n->idx != idx) || (n->left != NULL) || (n->right != NULL)) {
            nb_errors_1++;
        }
        res.push_back(test_eq(out, "create_node (no children)", ((nb_errors_1 == 0) && (nb_errors_2 == 0)), true));

        delete n;
    }

    for (int i = 0; i < T; i++) {
        node* nl = create_node(101);
        node* nr = create_node(1369);
        res.push_back(test_eq(out, "create_node (leaf, setting left to NULL)", nl->left == NULL, true));
        res.push_back(test_eq(out, "create_node (leaf, setting right to NULL)", nl->right == NULL, true));
        int c = Ndis(generator);
        int m = Ndis(generator);
        int idx = Ndis(generator);
        node* n = create_node(c, m, idx, nl, nr);
        // Test if we created the node correctly
        if ((n->c != c) || (n->m != m) || (n->idx != idx) ||
            (n->left != nl) || (n->right != nr)) {
            nb_errors_2++;
        }
        res.push_back(test_eq(out, "create_node (with children)", ((nb_errors_1 == 0) && (nb_errors_2 == 0)), true));
        delete n;
        delete nl;
        delete nr;
    }

    res.push_back(test_eq(out, "create_node", ((nb_errors_1 == 0) && (nb_errors_2 == 0)), true));

    return;
}

int test_Ex_5(std::ostream &out, const std::string test_name) {
    std::string entity_name = "kd-tree";
    start_test_suite(out, test_name);

    std::vector<int> res;
    test_create_node(5, out, res);
 
    return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}


void readTree(node*& p, std::ifstream& is) {
    int c;
    double m;
    int idx;
    is >> c;
    if (c == -1) {
        p = NULL;
        return;
    };
    is >> m >> idx;
    p = new node;
    p->c = c;
    p->m = m;
    p->idx = idx;
    readTree(p->left, is);
    readTree(p->right, is);
}

void test_defeatist(const std::string fname, bool verbose, const double eps,
                     std::ostream& out, std::vector<int>& res) 
{
    std::cout << std::endl << std::endl << std::endl;
    std::cout << "Testing the function defeatist()...\t\t\n";
    std::cout << "  Using data from the file " << fname << std::endl;

    std::ifstream in(fname, std::ios_base::in);

    int N;    // number of points
    int dim;  // dimension
    int T;    // number of tests
    in >> N;
    in >> dim;
    std::cout << "  " << N << " #points in dimension " << dim << std::endl;
    in >> T;
    std::cout << "  There are " << T << " tests \n";

    point P[10000];
    for (int i = 0; i < N; i++) {
        P[i] = new double[dim];
        for (int j = 0; j < dim; j++) {
            in >> P[i][j];
        }
    }
    node* tree;
    std::cout << "Building kd-tree..." << std::flush;
    std::cout << "  this is a precomputed kd-tree..." << std::flush;

    readTree(tree, in);

    // node* tree = build(P, 0, N, 0, dim);
    // if (tree == NULL) {
    //     std::cout << "\n Some functions are not implemented " << std::endl;
    //     std::cout << "[NOK]" << std::endl;
    //     return 0;
    // }
    std::cout << " done" << std::endl;

    point q = new double[dim];
    int nb_errors = 0;
    for (int t = 0; t < T; t++) {
        for (int j = 0; j < dim; j++) {
            in >> q[j];
        }
        int aidx;
        double adist;
        in >> aidx >> adist;

        double cdist = DBL_MAX;
        int cidx = -1;
        defeatist_search(tree, q, dim, P, cdist, cidx);

        if (verbose) {
            std::cout << std::endl
                      << "(" << t << ")  For the query point " << std::endl;
            print_point(q, dim);
            std::cout << "the NN has index : (actual, computed) = ( "
                      << aidx << ", " << cidx << ")" << endl;
            std::cout << " distance of q to NN (actual, computed) = ( "
                      << adist << ", " << cdist << ")" << endl;
        }
        res.push_back(test_eq_approx(out, "defeatist_search (distance to query point)", cdist, adist, eps));
	    res.push_back(test_eq(out, "defeatist_search (NN index)", cidx, aidx));
    }

    delete[] q;
    return;
}

int test_Ex_6(std::ostream &out, const std::string test_name) {
    std::string entity_name = "kd-tree";
    start_test_suite(out, test_name);

    std::vector<int> res;
    test_defeatist(default_path + "t_def.dat", false, deps, out, res);
    test_defeatist(default_path + "def-1000-100-100.dat", false, deps, out, res);

    return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}



void test_backtracking(const std::string fname, bool verbose, const double eps,
                     std::ostream& out, std::vector<int>& res) 
{
    std::cout << std::endl << std::endl << std::endl;
    std::cout << "Testing the function backtracking_search()...\t\t\n";
    std::cout << "  Using data from the file " << fname << std::endl;

    std::ifstream in(fname, std::ios_base::in);

    int N;    // number of points
    int dim;  // dimension
    int T;    // number of tests
    in >> N;
    in >> dim;
    std::cout << "  " << N << " #points in dimension " << dim << std::endl;
    in >> T;
    std::cout << "  There are " << T << " tests \n";

    point P[10000];
    for (int i = 0; i < N; i++) {
        P[i] = new double[dim];
        for (int j = 0; j < dim; j++) {
            in >> P[i][j];
        }
    }
    std::cout << "Building kd-tree..." << std::flush;
    node* tree = build(P, 0, N, 0, dim);
    // Did we build the tree correctly?
    bool b1 = test_eq(out, "backtracking_search (in building the tree)", (tree != NULL), true);
    std::cout << " done" << std::endl;

    point q = new double[dim];
    int nb_errors = 0;
    for (int t = 0; t < T; t++) {
        for (int j = 0; j < dim; j++) {
            in >> q[j];
        }
        int aidx;
        double adist;
        in >> aidx >> adist;

        double cdist = DBL_MAX;
        int cidx = -1;

        backtracking_search(tree, q, dim, P, cdist, cidx);

        int Lidx = linear_scan(q, dim, P, N);

        if (verbose) {
            std::cout << std::endl
                      << "(" << t << ")  For the query point " << std::endl;
            print_point(q, dim);
            std::cout << "the NN is at distance : (actual, computed) = ( "
                      << adist << ", " << cdist << ") " << endl;
            std::cout << "           with index : (actual, computed) = ( "
                      << aidx << ", " << cidx << ")" << endl;
        }
        // reduntant tests. TODO: find a better way.
        res.push_back(b1 && 
        test_eq_approx(out, "backtracking_search (distance to query point)", cdist, adist, eps)
        && test_eq(out, "backtracking_search (NN index)", cidx, aidx));

    }

    delete[] q;
    return;
}

int test_Ex_7(std::ostream &out, const std::string test_name) {
    std::string entity_name = "kd-tree";
    start_test_suite(out, test_name);
    
    std::vector<int> res;
    test_backtracking(default_path + "t_back.dat", false, deps, out, res);
    test_backtracking(default_path + "back-1000-100-100.dat", false, deps, out, res);

    return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}


bool compare_search_algorithms(const int T, const double eps ) {

    const int dim = 4;  // dimension (hard-coded)
    int n = 10000;      // upper bound on number of data points in R^{dim}
    int nt = 1000;      // nt query points

    point P[10000];
    std::string itypes[10000];

    // Read in the arguments
 //   print_help_msg();
   // int arg = 1;
   // int T = (argc > arg) ? std::stoi(argv[arg]) : dT;
    std::cout << std::endl << std::endl << std::endl;
    std::ifstream is("./grading/tests/iris2.dat");
    assert(is.is_open());
    for (int k = 0; k < n; k++) {
        P[k] = new double[dim];
        for (int i = 0; i < dim; i++) {
            is >> P[k][i];
        }
        is >> itypes[k];
    }
    std::cout << "Done reading the iris database! ";
    std::cout << "There are " << n << " observations. \n\n";

    std::cout << "Building kd-tree..." << std::flush;
    node* tree = build(P, 0, n, 0, dim);
    if (tree == NULL) {
        std::cout << "\n Some functions are not implemented " << std::endl;
        std::cout << "[NOK]" << std::endl;
        return 0;
    }
    std::cout << " done" << std::endl;

    cout << "We perform " << T << " tests" << endl << endl;
    // Random query points
    point* q = new point[T];
    for (int i = 0; i < T; i++) {
        q[i] = new double[dim];
        for (int j = 0; j < dim; j++) {
            q[i][j] = 10 * (double)rand() / RAND_MAX;
        }
    }

    std::clock_t start, end;

    // Time the linear_scan algorithm
    int idx_ls[T];
    std::cout << "Benchmarking linear scan... " << std::flush;
    start = std::clock();
    for (int i = 0; i < T; i++) {
        idx_ls[i] = linear_scan(q[i], dim, P, n);
    }
    end = std::clock();
    cout << "\n   Total time: " << (((float)(end - start)) )
         << "\t\tavg time: " << (float((end - start) / T))  
         << " us" << endl
         << endl
         << endl;

    // Time the defeatist algorithm
    int idx_def[T];
    int def_errors = 0;
    std::cout << "Benchmarking defeatist..." << std::flush;
    start = std::clock();
    for (int i = 0; i < T; i++) {
        double dist_qP = DBL_MAX;
        defeatist_search(tree, q[i], dim, P, dist_qP, idx_def[i]);
        if (idx_def[i] != idx_ls[i]) {
            def_errors++;
        }
    }
    end = std::clock();
    cout << "\n   Total time: " << (((float)(end - start))  )
         << "\t\tavg time: " << (float((end - start) / T)) 
         << " us" << endl;
    cout << "   #Errors = " << def_errors
         << "\t accuracy: " << 100.0 * (T - def_errors) / T << "% \n";
    cout << endl
         << endl;

    // Time the backtracking algorithm
    int idx_bac[T];
    int bac_errors = 0;
    std::cout << "Benchmarking backtracking..." << std::flush;
    start = std::clock();
    for (int i = 0; i < T; i++) {
        double dist_qP = DBL_MAX;
        backtracking_search(tree, q[i], dim, P, dist_qP, idx_bac[i]);
        if (fabs(dist_qP - dist(q[i], P[idx_ls[i]], dim)) > eps) {
        // if (idx_bac[i] != idx_ls[i]) {
            // cout << "D : " << dist(q[i], P[idx_bac[i]], dim) <<  " " <<  dist(q[i], P[idx_ls[i]], dim) << endl;;
            bac_errors++;

            return 0;
        }
    }
    end = std::clock();
    cout << "\n   Total time: " << (((float)(end - start)))
         << "\t\tavg time: " << (float((end - start) / T)) 
         << " us" << std::endl;
    cout << "   #Errors = " << bac_errors
         << "\t accuracy: " << 100. * (T - bac_errors) / T << "% \n";

    return true;
}


int test_Ex_8(std::ostream &out, const std::string test_name) {
    std::string entity_name = "kd-tree";
    start_test_suite(out, test_name);

    std::vector<int> res = {
        test_eq(out, "compare_algorithms_one", compare_search_algorithms(100, deps), true),
        test_eq(out, "compare_algorithms_one", compare_search_algorithms(1000, deps), true)
        //,test_eq(out, "compare_algorithms_one", compare_search_algorithms(2000, deps), true )
    };

    return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}

//-------------------------------------------------------------------



int grading(std::ostream &out, const int test_case_number)
{
/**

Annotations used for the autograder.

[START-AUTOGRADER-ANNOTATION]
{
  "total" : 8,
  "names" : [
      "retrieval.cpp::test_Ex_1", 
      "retrieval.cpp::test_Ex_2", 
      "retrieval.cpp::test_Ex_3", 
      "retrieval.cpp::test_Ex_4", 
      "retrieval.cpp::test_Ex_5",
      "retrieval.cpp::test_Ex_6",
      "retrieval.cpp::test_Ex_7",
      "retrieval.cpp::test_Ex_8"    
        ],
  "points" : [10, 10, 16, 16, 16, 16, 16, 0]
}
[END-AUTOGRADER-ANNOTATION]
*/

    int const total_test_cases = 8;
    std::string const test_names[total_test_cases] = {"Test_Ex_1", "Test_Ex_2",  
                    "Test_Ex_3", "Test_Ex_4", "Test_Ex_5", 
                    "Test_Ex_6", "Test_Ex_7", "Test_Ex_8"};
    int const points[total_test_cases] = {10, 10, 16, 16, 16, 16, 16, 0};
    int (*test_functions[total_test_cases]) (std::ostream &, const std::string) = {
      test_Ex_1, test_Ex_2, test_Ex_3, test_Ex_4, test_Ex_5, test_Ex_6, test_Ex_7, test_Ex_8
    };

    return run_grading(out, test_case_number, total_test_cases,
                       test_names, points,
                       test_functions);
}

} // End of namepsace tdgrading
