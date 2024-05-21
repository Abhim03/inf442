#pragma once

#include <iostream>

class Node
{
private:
    // Forward-propagated signal
    double signal;

    // Back-propagated error
    // Must be initialised to 0 (see comment to step_back() )
    double back_value;

    // For testing only
    static int count;
public:
    Node();
    Node(double _signal);
    ~Node();

    double get_signal() const;
    double get_back_value() const;

    void set_signal(double _signal);
    void set_back_value(double _back_value);

    static int get_count();
};

std::ostream &operator<<(std::ostream &str, const Node &n);
