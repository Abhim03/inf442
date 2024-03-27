#include "edge.hpp"  // This is the header for the class implemented here

#include <cassert> // This provides the assert methods
#include <cstddef> // This provides NULL

/* edge -- method implementations */

int edge::count_compare = 0;

edge::edge() {
    p1 = 0;
    p2 = 0;
    length = 0;
}

edge::edge(int _p1, int _p2, double _length) {
    assert(_p1 >= 0);
    assert(_p2 >= 0);

    p1 = _p1;
    p2 = _p2;
    length = _length;
}

edge::~edge() {
    edge::count_compare = 0;
}

double edge::get_length() const {
    return length;
}

bool edge::compare(const edge& e1, const edge& e2) {
    count_compare++; // for testing only
    return e1.get_length() < e2.get_length();
}

int edge::get_count_compare() {
    return count_compare;
}

int edge::get_p1() const {
    return p1;
}

int edge::get_p2() const {
    return p2;
}
