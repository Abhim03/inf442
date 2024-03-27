#pragma once

#include "cloud.hpp"

/* edge -- pairs of cloud point indices with methods to
 *  - compute the distance from source to target
 *  - compare the lengths of two edges -- needed for sorting
 */

class edge {
    int p1, p2;
    double length;

    static int count_compare; // For testing only

public:
    edge();
    edge(int _p1, int _p2, double _length);
    ~edge();
    
    int get_p1() const;
    int get_p2() const;
    double get_length() const;

    static bool compare(const edge& e1, const edge& e2);
    static int get_count_compare();
};
