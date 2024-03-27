#include <algorithm>
#include <cmath>
#include <numeric>

#include "../gradinglib/gradinglib.hpp"
#include "../point.hpp"
#include "../cloud.hpp"
#include "../graph.hpp"
#include "../dendrogram.hpp"

namespace tdgrading {

using namespace testlib;
using namespace std;

template <typename T1, typename T2, typename T3, typename... Arguments>
bool test_sorted(std::ostream &out,
                 const std::string &function_name,
                 T1 array_start,
                 T2 array_length,
                 T3 comparison,
                 const Arguments&... args) {

    bool success = std::is_sorted(array_start, array_start + array_length, comparison);
    out << (success ? (SUCCESS + " ") : (FAILURE + " "));

    print_tested_function(out, function_name, args...);
    out << std::endl;

    return success;
}

//-----------------------------------------------------------------------------

int test_2q1(std::ostream &out, const std::string test_name) {
    std::string entity_name = "point";
    start_test_suite(out, test_name);
    std::vector<int> res;

    point p;
    res.push_back(test_eq(out, "set_dim", p.set_dim(3), true));
    res.push_back(test_eq(out, "get_dim", p.get_dim(), 3));
    res.push_back(test_eq(out, "set_dim", p.set_dim(5), false));
    res.push_back(test_eq(out, "set_dim", p.set_dim(3), false));

    return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}

//-----------------------------------------------------------------------------

int test_3q1(std::ostream &out, const std::string test_name) {
    std::string entity_name = "graph";
    start_test_suite(out, test_name);
    std::vector<int> res;

    print(out, "Testing creating a graph from a cloud\n");

    cloud c(2, 4);
    point p1, p2, p3, p4;
    p1.set_dim(2);
    p2.set_dim(2);
    p2.coords[0] = 1.;
    p3.set_dim(2);
    p3.coords[0] = 1.;
    p3.coords[1] = 0.5;
    p4.set_dim(2);
    p4.coords[0] = 1.;
    p4.coords[1] = 2.;
    c.add_point(p1);
    c.add_point(p2);
    c.add_point(p3);
    c.add_point(p4);

    graph G(c);
    res.push_back(test_eq(out, "get_num_edges", G.get_num_edges(), 6));
    res.push_back(test_eq(out, "get_num_nodes", G.get_num_nodes(), 4));

    double const edge_lengths[6] = {0.5, 1., 1.118, 1.5, 2, 2.236};
    for (int i = 0; i < 6; ++i) {
        res.push_back(test_eq_approx(out, "edge_length", G.get_edge(i)->get_length(), edge_lengths[i], 0.01));
    }

    res.push_back(test_sorted(out, "edges are sorted", G.get_edges(), G.get_num_edges(), edge::compare));

    print(out, "\nTesting creating a graph from a matrix\n");

    int n = 3;
    std::string names[3] = {"A", "B", "C"};
    double** dist_mat = new double*[n];
    for (int i = 0; i < n; ++i)
        dist_mat[i] = new double[n];

    dist_mat[0][0] = 0.;
    dist_mat[0][1] = 1.;
    dist_mat[0][2] = 2.;
    dist_mat[1][0] = 1.;
    dist_mat[1][1] = 0.;
    dist_mat[1][2] = 3.;
    dist_mat[2][0] = 2.;
    dist_mat[2][1] = 3.;
    dist_mat[2][2] = 0.;

    graph G2(n, names, dist_mat);
    res.push_back(test_eq(out, "get_num_edges", G2.get_num_edges(), 3));
    res.push_back(test_eq(out, "get_num_nodes", G2.get_num_nodes(), 3));
    for (int i = 0; i < 3; ++i) {
        res.push_back(test_eq(out, "get_name", G2.get_name(i), names[i]));
    }
    double const edge_lengths2[3] = {1., 2., 3.};
    for (int i = 0; i < 3; ++i) {
        res.push_back(test_eq_approx(out, "edge_length", G2.get_edge(i)->get_length(), edge_lengths2[i], 0.01));
    }

    res.push_back(test_sorted(out, "edges are sorted", G2.get_edges(), G2.get_num_edges(), edge::compare));

    return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}

//-----------------------------------------------------------------------------

int test_3q2(std::ostream &out, const std::string test_name) {
    std::string entity_name = "graph";
    start_test_suite(out, test_name);
    std::vector<int> res;

    cloud c(2, 4);
    point p1, p2, p3, p4;
    p1.set_dim(2);
    p2.set_dim(2);
    p2.coords[0] = 1.;
    p3.set_dim(2);
    p3.coords[0] = 1.;
    p3.coords[1] = 0.5;
    p4.set_dim(2);
    p4.coords[0] = 1.;
    p4.coords[1] = 2.;
    c.add_point(p1);
    c.add_point(p2);
    c.add_point(p3);
    c.add_point(p4);

    graph G(c);

    double const edge_lengths[6] = {0.5, 1., 1.118, 1.5, 2, 2.236};
    G.start_iteration();
    int pos = 0;
    edge* cur = G.get_next();
    while (cur != NULL) {
        res.push_back(test_eq_approx(out, "edge_length", cur->get_length(), edge_lengths[pos], 0.01));
        ++pos;
        cur = G.get_next();
    }

    return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}


//-----------------------------------------------------------------------------

void _test_find(std::ostream& out, const cloud& c, std::vector<int>& res) {
    print(out, "Testing find\n");

    graph G(c);
    dendrogram d(G);
    int find[] = {0, 1, 2, 3, 4};
    int parent[] = {-1, -1, -1, -1, -1};

    for (int i = 0; i < d.get_n(); i++)
        res.push_back(test_eq(out, "find", d.find(i), find[i]));

    parent[0] = 1;
    d.set_parent(parent);
    find[0] = 1;
    for (int i = 0; i < d.get_n(); i++)
        res.push_back(test_eq(out, "find", d.find(i), find[i]));

    parent[2] = 3;
    d.set_parent(parent);
    find[2] = 3;
    for (int i = 0; i < d.get_n(); i++)
        res.push_back(test_eq(out, "find", d.find(i), find[i]));

    parent[1] = 2;
    d.set_parent(parent);
    find[0] = 3;
    find[1] = 3;
    for (int i = 0; i < d.get_n(); i++)
        res.push_back(test_eq(out, "find", d.find(i), find[i]));

    parent[4] = 3;
    d.set_parent(parent);
    find[4] = 3;
    for (int i = 0; i < d.get_n(); i++)
        res.push_back(test_eq(out, "find", d.find(i), find[i]));
}

// For getting merge out
class test_dendrogram: public dendrogram {
public:
    test_dendrogram(graph& _g) : dendrogram (_g) {}

    void merge (int p1, int p2, double d) {
        dendrogram::merge(new edge(p1, p2, d));
    }
};

void _test_merge(std::ostream& out, const cloud& c, std::vector<int>& res) {
    print(out, "Testing merge\n");

    graph G(c);
    test_dendrogram d(G);

    d.merge(1, 0, 1);
    d.merge(3, 2, 1);
    d.merge(2, 1, 2);
    d.merge(4, 3, 5);

    int parent[] = {  1,  3,   3,  -1,   3};
    int rank[]   = {  0,  1,   0,   2,   0};
    int left[]   = { -1,  2,  -1,  -1,   1};
    int down[]   = { -1,  0,  -1,   4,  -1};
    double height[] = {0.5,  1, 0.5,  -1, 2.5};

    for (int i = 0; i < d.get_n(); ++i)
        res.push_back(test_eq(out, "parent", d.get_parent(i), parent[i]));

    for (int i = 0; i < d.get_n(); ++i)
        res.push_back(test_eq(out, "left", d.get_left(i), left[i]));

    for (int i = 0; i < d.get_n(); ++i)
        res.push_back(test_eq(out, "down", d.get_down(i), down[i]));

    for (int i = 0; i < d.get_n(); ++i)
        res.push_back(test_eq(out, "rank", d.get_rank(i), rank[i]));

    for (int i = 0; i < d.get_n(); ++i)
        res.push_back(test_eq_approx(out, "height", d.get_height(i), height[i], 0.01));
}

void _test_build(std::ostream& out, const cloud& c, std::vector<int>& res) {
    print(out, "Testing build\n");

    graph G(c);
    dendrogram d(G);
    d.build();

    int parent[] = {1, 3, 3, -1, 3};
    int rank[] = {0, 1, 0, 2, 0};
    int left[] = {-1, 2, -1, -1, 1};
    int down[] = {-1, 0, -1, 4, -1};
    double height[] = {0.5,  1, 0.5,  -1, 2.5};

    for (int i = 0; i < 5; ++i) {
        res.push_back(test_eq(out, "parent", d.get_parent(i), parent[i]));
        res.push_back(test_eq(out, "rank", d.get_rank(i), rank[i]));
        res.push_back(test_eq(out, "down", d.get_down(i), down[i]));
        res.push_back(test_eq(out, "left", d.get_left(i), left[i]));
        res.push_back(test_eq_approx(out, "height", d.get_height(i), height[i], 0.01));
   }
}

void _test_set_clusters(std::ostream& out, const cloud& c, std::vector<int>& res) {
    print(out, "Testing set_clusters\n");

    graph G(c);
    dendrogram d(G);
    d.build();

    int clusters[] = {-1, -1, -1, -1, -1};
    for (int i = 0; i < d.get_n(); ++i)
        res.push_back(test_eq(out, "set_clusters", d.get_cluster(i), clusters[i]));

    d.clear_clusters();
    d.set_clusters(0.5);
    clusters[0] = 1;
    clusters[1] = 1;
    clusters[2] = 3;
    clusters[3] = 3;
    clusters[4] = 4;
    for (int i = 0; i < d.get_n(); ++i)
        res.push_back(test_eq(out, "set_clusters", d.get_cluster(i), clusters[i]));

    d.clear_clusters();
    d.set_clusters(0.7);
    for (int i = 0; i < d.get_n(); ++i)
        res.push_back(test_eq(out, "set_clusters", d.get_cluster(i), clusters[i]));

    d.clear_clusters();
    d.set_clusters(1.0);
    clusters[0] = 3;
    clusters[1] = 3;
    clusters[2] = 3;
    clusters[3] = 3;
    clusters[4] = 4;
    for (int i = 0; i < d.get_n(); ++i)
        res.push_back(test_eq(out, "set_clusters", d.get_cluster(i), clusters[i]));

    d.clear_clusters();
    d.set_clusters(2.5);
    clusters[4] = 3;
    for (int i = 0; i < d.get_n(); ++i)
        res.push_back(test_eq(out, "set_clusters", d.get_cluster(i), clusters[i]));
}

void _test_count_clusters(std::ostream& out, const cloud& c, std::vector<int>& res) {
    print(out, "Testing set_clusters\n");

    graph G(c);
    dendrogram d(G);
    d.build();

    d.clear_clusters();
    d.set_clusters(0.5);
    res.push_back(test_eq(out, "count_ns_clusters", d.count_ns_clusters(), 2));

    d.clear_clusters();
    d.set_clusters(0.7);
    res.push_back(test_eq(out, "count_ns_clusters", d.count_ns_clusters(), 2));

    d.clear_clusters();
    d.set_clusters(1.0);
    res.push_back(test_eq(out, "count_ns_clusters", d.count_ns_clusters(), 1));

    d.clear_clusters();
    d.set_clusters(2.5);
    res.push_back(test_eq(out, "count_ns_clusters", d.count_ns_clusters(), 1));
}

bool __test_heights(std::ostream& out, std::vector<int>& res, dendrogram& d, int* clusters, double* height) {
    int clusters_ans[d.get_n()];
    double height_ans[d.get_n()];
    for (int i = 0; i < d.get_n(); ++i) {
        clusters_ans[i] = d.get_cluster(i);
    	height_ans[i] = 0.;
	    if (i == d.get_cluster(i)) {
	        height_ans[i] = d.get_cluster_height(i);
    	}
    }
    if (std::equal(clusters_ans, clusters_ans, clusters)) {
        for (int i = 0; i < d.get_n(); ++i) {
            if (fabs(height_ans[i] - height[i]) > 0.01) {
                res.push_back(test_eq(out, "cluster heights are correct", 0, 1));
                return false;
            }
        }
        return true;
    }
    return false;
}

void _test_cluster_height(std::ostream& out, const cloud& c, std::vector<int>& res) {
    print(out, "Testing get_cluster_height\n");

    graph G(c);
    dendrogram d(G);
    d.build();

    d.clear_clusters();
    print(out, "Checking clusters at height 0.5\n");
    d.set_clusters(0.5);
    // possible correct answers
    double height[] = {0,  0.5, 0,  0.5, 0};
    int clusters[] = {1, 1, 3, 3, 4};
    double height_alt[] = {0.5, 0, 0.5, 0, 0};
    int clusters_alt[] = {0, 0, 2, 2, 4};

    if (!__test_heights(out, res, d, clusters, height) && !__test_heights(out, res, d, clusters_alt, height_alt)) {
        res.push_back(test_eq(out, "Clusters are correct", 0, 1));
    }

    print(out, "Checking clusters at height 1\n");
    d.clear_clusters();
    d.set_clusters(1);
    height[1] = 0;
    height[3] = 1;
    clusters[0] = 3;
    clusters[1] = 3;
    height_alt[0] = 1;
    height_alt[2] = 0;
    clusters_alt[2] = 0;
    clusters_alt[3] = 0;
    if (!__test_heights(out, res, d, clusters, height) && !__test_heights(out, res, d, clusters_alt, height_alt)) {
        res.push_back(test_eq(out, "Clusters are correct", 0, 1));
    }

    print(out, "Checking clusters at height 2.5\n");
    d.clear_clusters();
    d.set_clusters(2.5);
    height[3] = 2.5;
    clusters[4] = 3;
    height_alt[0] = 2.5;
    clusters_alt[4] = 0;
    if (!__test_heights(out, res, d, clusters, height) && !__test_heights(out, res, d, clusters_alt, height_alt)) {
        res.push_back(test_eq(out, "Clusters are correct", 0, 1));
    }
    res.push_back(1);
}

//-----------------------------------------------------------------------------

int run_dendro_tests(std::ostream& out, const std::string test_name, void func(std::ostream&, const cloud&, std::vector<int>&)) {
    std::string entity_name = "dendrogram";
    start_test_suite(out, test_name);
    std::vector<int> res;

    cloud c (1, 6);
    double points[] = {0.0, 1.0, 3.0, 4.0, 9.0};
    std::string labels[] = {"0", "1", "3", "4", "9"};
    point p;
    p.set_dim(1);
    for (int i = 0; i < 5; ++i) {
        p.coords[0] = points[i];
        p.name = labels[i];
        c.add_point(p);
    }

    func(out, c, res);

    return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}

//-----------------------------------------------------------------------------

int test_3q4(std::ostream &out, const std::string test_name) {
    return run_dendro_tests(out, test_name, &_test_find);
}

//-----------------------------------------------------------------------------

int test_3q5(std::ostream &out, const std::string test_name) {
    return run_dendro_tests(out, test_name, &_test_merge);
}

//-----------------------------------------------------------------------------

int test_3q6(std::ostream &out, const std::string test_name) {
    return run_dendro_tests(out, test_name, &_test_build);
}

//-----------------------------------------------------------------------------

int test_4q1(std::ostream &out, const std::string test_name) {
    return run_dendro_tests(out, test_name, &_test_set_clusters);
}

//-----------------------------------------------------------------------------

int test_4q2(std::ostream &out, const std::string test_name) {
    return run_dendro_tests(out, test_name, &_test_count_clusters);
}

//-----------------------------------------------------------------------------

int test_4q3(std::ostream &out, const std::string test_name) {
    return run_dendro_tests(out, test_name, &_test_cluster_height);
}

//-----------------------------------------------------------------------------

int grading(std::ostream &out, const int test_case_number)
{
/**

Annotations used for the autograder.

[START-AUTOGRADER-ANNOTATION]
{
  "total" : 9,
  "names" : [
      "point.cpp::test_2q1",
      "graph.cpp::test_3q1",
      "graph.cpp::test_3q2",
      "dendrogram.cpp::test_3q4",
      "dendrogram.cpp::test_3q5",
      "dendrogram.cpp::test_3q6",
      "dendrogram.cpp::test_4q1",
      "dendrogram.cpp::test_4q2",
      "dendrogram.cpp::test_4q3"
  ],
  "points" : [11, 11, 11, 11, 11, 11, 11, 11, 12]
}
[END-AUTOGRADER-ANNOTATION]
*/

    int const total_test_cases = 9;
    std::string const test_names[total_test_cases] = {"Test_2q1", "Test_3q1", "Test_3q2", "Test_3q4", "Test_3q5", "Test_3q6", "Test_4q1", "Test_4q2", "Test_4q3"};
    int const points[total_test_cases] = {11, 11, 11, 11, 11, 11, 11, 11, 12};
    int (*test_functions[total_test_cases]) (std::ostream &, const std::string) = {
        test_2q1, test_3q1, test_3q2, test_3q4, test_3q5, test_3q6, test_4q1, test_4q2, test_4q3
    };

    return run_grading(out, test_case_number, total_test_cases,
                       test_names, points,
                       test_functions);
}

} // End of namepsace tdgrading
