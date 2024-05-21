#include "node.hpp"

#include <cassert>
#include <iostream>

Node::Node() : signal(0), back_value(0) {
    count++;
}
Node::Node(double _signal) : signal(_signal), back_value(0) {
    count++;
}
Node::~Node() {
    count--;
}

double Node::get_signal() const
{
    return signal;
}

double Node::get_back_value() const
{
    return back_value;
}

void Node::set_signal(double _signal)
{
    assert(this != (void *)0xffffffffffffffff);
    signal = _signal;
}

void Node::set_back_value(double _back_value)
{
    back_value = _back_value;
}

int Node::count = 0;

int Node::get_count() {
    return count;
}

std::ostream &operator<<(std::ostream &str, const Node &n)
{
    return str << "(" << n.get_signal() << ", " << n.get_back_value() << ")";
}
