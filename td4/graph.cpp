#include "graph.hpp" // This is the header for the class implemented here

#include "cloud.hpp" // Used in the constructor
#include "edge.hpp"  // Used in almost all methods

#include <algorithm> // This provides the sort() method for the constructor
#include <cassert>
#include <iostream>
#include <sstream>

/* graph -- method implementations */

graph::graph(const cloud &_c) {
    n = _c.get_n();
    size = n * (n - 1) / 2;
    // TODO: Exercise 2

    iterator_pos = 0;
}

graph::graph(long _n, const std::string _node_names[], double **dist_matrix) {
    n = _n;
    size = n * (n - 1) / 2;
    // TODO: Exercise 2

    iterator_pos = 0;

}

graph::~graph() {
    // TODO: Exercise 2
}

long graph::get_num_edges() const {
    // TODO: Exercise 2

    return 0;
}

std::string &graph::get_name(int i) const {
    assert(i >= 0 && i < n);
    return node_names[i];
}

edge *graph::get_edge(long i) {
    return &edges[i];
}

edge *graph::get_edges() {
    return edges;
}

long graph::get_num_nodes() const {
    return n;
}

void graph::start_iteration() {
    // TODO: Exercise 3
}

edge *graph::get_next() {
    // TODO: Exercise 3

    return NULL;
}

graph *graph::load_matrix(std::ifstream &is) {
    assert(is.is_open());

    std::string buffer;
    getline(is, buffer, '\n');
    int n = std::stoi(buffer);
    std::string node_names[n];
    for (size_t i = 0; i < n; ++i) {
	getline(is, node_names[i], '\n');
    }

    std::cout << "Names read" << std::endl;
    double **dist_matrix = new double*[n];
    for (int i = 0; i < n; ++i) {
        dist_matrix[i] = new double[n];
	std::getline(is, buffer, '\n');
        std::stringstream ls;
	ls << buffer;
	std::string dist_str;
	for (int j = 0; j < n; ++j) {
	    std::getline(ls, dist_str, ',');
	    dist_matrix[i][j] = std::stod(dist_str);
	}
    }

    graph *g = new graph(n, node_names, dist_matrix);
    for (size_t i = 0; i < n; ++i) {
        delete[] dist_matrix[i];
    }
    delete[] dist_matrix;

    return g;
}
