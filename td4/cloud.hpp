#pragma once // single inclusion

#include <fstream>

#include "point.hpp"

class cloud {
    int n;

    // Maximum possible number of points
    int nmax;

    point *points;

public:
    cloud(int _d, int _nmax);
    ~cloud();

    // Getters

    int get_n() const;
    const point &get_point(int i) const;

    // Helper methods
    void add_point(point &p);
    void load(std::ifstream &is);
};
