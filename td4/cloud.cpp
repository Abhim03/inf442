#include "cloud.hpp" // The header for the class implemented here
#include "point.hpp" // Used in all methods

#include <cassert> // This provides the assert() method
#include <iostream>
#include <sstream>

cloud::cloud(int _d, int _nmax) {
    point::set_dim(_d);

    nmax = _nmax;
    n = 0;

    points = new point[nmax];
}

cloud::~cloud() {
    delete[] points;
}

int cloud::get_n() const {
    return n;
}

const point &cloud::get_point(int i) const {
    return points[i];
}

void cloud::add_point(point &p) {
    assert(n < nmax);

    for (int m = 0; m < point::get_dim(); m++) {
        points[n].coords[m] = p.coords[m];
        points[n].name = p.name;
    }

    n++;
}

void cloud::load(std::ifstream &is) {
    assert(is.is_open());

    // Point to read into
    point p;

    std::string line;
    // Skipping header
    std::getline(is, line, '\n');
    while (std::getline(is, line, '\n')) {
        std::stringstream ls;
	ls << line;
	std::string scoord;
	for (int i = 0; i < point::get_dim(); ++i) {
	    std::getline(ls, scoord, ',');
	    p.coords[i] = std::stod(scoord);
	}
        std::getline(ls, p.name, ',');
	add_point(p);
    }
}
