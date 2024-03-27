
#include "point.hpp"
#include "cloud.hpp"
#include "dendrogram.hpp"

#include <cstdlib> // for rand, srand
#include <ctime>   // for time
#include <cassert>
#include <cmath>   // for abs

#include <chrono>
#include <iostream>
using std::cout;
using std::endl;

void print_point(dendrogram *d, int i) {
    cout << i << " (" << d->get_name(i) << ")";
}

int trace_find(dendrogram *d, int i) {
    int p = d->get_parent(i);

    cout << d->get_rank(i) << ", ";
    print_point(d, i);
    cout << ", ";
    if (p != -1) {
        print_point(d, p);
        cout << ", " << d->get_height(i);
    }
    cout << endl;

    if (p == -1)
        return i;

    return trace_find(d, p);
}

void init_cloud(cloud &c, double *points, std::string *labels, int size) {
    cout << "with points ";
    // temporary container
    point p;
    for (int i = 0; i < size; i++) {
        cout << points[i] << " ";
        p.coords[0] = points[i];
        p.name = labels[i];
        c.add_point(p);
    }
    cout << "\t";
}

int main(int argc, char *argv[]) {
    srand(time(NULL));

    if (argc < 2) {
        cout << "Usage (clouds): " << endl
             << argv[0] << " " << "<file name> <data dimension> <nb points>" << endl
             << "Example: " << argv[0] << " ./csv/iris.csv 4 150" << endl << endl
             << "Usage (distance matrix):" << argv[0] << " "
             << "<file name>" << endl
             << "Example: " << argv[0] << " ./csv/languages.csv" << endl << endl;
        return 0;
    }

    std::ifstream is(argv[1]);
    if (is.is_open()) {
        graph* g;
        if (argc == 4) {
            int d = std::stoi(argv[2]);
            int nmax = std::stoi(argv[3]);

            cloud c(d, nmax);

            c.load(is);

            cout << "Loaded "
                << c.get_n()
                << " points from "
                << argv[1];
            cout << ((c.get_n() == nmax) ? "\t[OK]" : "\t[NOK]") << endl;
            if (c.get_n() != nmax)
                cout << "Must have loaded " << nmax << " points" << endl;
            g = new graph(c);
        } else {
            g = graph::load_matrix(is);
            cout << "Loaded "
                 << g->get_num_nodes()
                 << " vertices from "
                 << argv[1] << endl;    
        }
        is.close();

        auto start = std::chrono::high_resolution_clock::now();
        dendrogram dg(*g);
        dg.build();
	auto finish = std::chrono::high_resolution_clock::now();
	auto runtime = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);
	std::cout << "Execution time: " << runtime.count() << std::endl;

        cout << "Height of the dendrogram:\t"
            << dg.get_dendro_height() << endl
            << "(For iris.data, height should be 0.820061)" << endl;

        cout << "Printing traces to root from 10 random points..." << endl;
        cout << "Rank, point, parent, height" << endl;
        srand(static_cast<unsigned>(time(0)));
        for (int i = 0; i < 10; i++) {
            int node = rand() % dg.get_n();
            trace_find(&dg, node);
            cout << endl;
        }

        double eps = 0.01;
        cout << "Looking for significant heights (up to " << eps << ")...";
        dg.find_heights(eps);
        cout << "\tdone" << endl;
        int count = dg.get_count_sign_heights();
        cout << "Number of significant heights (up to "
            << eps << ") = " << count << ": " << endl;
        for (int i = 0; i < count; i++)
            cout << dg.get_sign_height(i) << " ";
        cout << endl;

        cout << "Printing clusters at significant heights" << endl;
        for (int i = 0; i < count; i++) {
            dg.clear_clusters();
            dg.set_clusters(dg.get_sign_height(i));
            cout << dg.count_ns_clusters()
                << " cluster"
                << (dg.count_ns_clusters() > 1 ? "s" : "")
                << " found at height " << dg.get_sign_height(i) << endl;
            dg.print_clusters();
            dg.print_dendrogram();
            cout << endl;
        }
    }

    return 0;
}
