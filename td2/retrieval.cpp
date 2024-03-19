#include <algorithm>  // for sort
#include <cassert>    // for assertions
#include <cfloat>     // for DBL_MAX
#include <cmath>      // for math operations like sqrt, log, etc
#include <cstdlib>    // for rand, srand
#include <ctime>      // for clock
#include <fstream>    // for ifstream
#include <iostream>   // for cout

#include "retrieval.hpp"

using std::cout;
using std::endl;

/*****************************************************
 * TD 2: K-Dimensional Tree (kd-tree)                *
 *****************************************************/

void print_point(point p, int dim) {
    std::cout << "[ ";
    for (int i = 0; i < dim; ++i) {
        std::cout << p[i] << " ";
    }
    std::cout << "]" << std::endl;
}

void pure_print(point p, int dim) {
    std::cout << p[0];
    for (int j = 1; j < dim; j++)
        std::cout << " " << p[j];
    std::cout << "\n";
}

/*****************************************************
 * Exercise 1: dist                                  *
 *****************************************************/
/**
 * This function computes the Euclidean distance between two points
 *
 * @param p  the first point
 * @param q the second point
 * @param dim the dimension of the space where the points live
 * @return the distance between p and q, i.e., the length of the segment pq
 */
double dist(point p, point q, int dim) {
    // Exercise 1

    double d=0;
    for(int i=0;i<dim;i++){
        d+= pow(p[i]-q[i],2);

    }

    return std:: sqrt(d);

}

/*****************************************************
 * Exercise 2: linear_scan                           *
 *****************************************************/
/**
 * This function for a given query point q  returns the index of the point
 * in an array of points P that is closest to q using the linear scan algorithm.
 *
 * @param q the query point
 * @param dim the dimension of the space where the points live
 * @param P a set of points
 * @param n the number of points in P
 * @return the index of the point in p that is closest to q
 */
int linear_scan(point q, int dim, point* P, int n) {
    // Exercise 2
    int index = 0;
    for(int j = 0; j < n; j++) {
        if(dist(q, P[j], dim) < dist(q, P[index], dim)) {
            index = j;
        } else if (dist(q, P[j], dim) == dist(q, P[index], dim)) {
            if(j <= index) {
                index = j;
            }
        }
    }
    return index;
}
 

/*****************************************************
 * Exercise 3: compute_median                        *
 *****************************************************/
/**
 * This function computes the median of all the c coordinates 
 * of an subarray P of n points that is P[start] .. P[end - 1]
 *
 * @param P a set of points
 * @param start the starting index
 * @param end the last index; the element P[end] is not considered
 * @return the median of the c coordinate
 */
double compute_median(point* P, int start, int end, int c) {
    double c_thvalue[end-start];
    double median = 0.0;
    for(int i=0;i<end-start;i++){
        c_thvalue[i]=P[i+start][c];
    }
    std::sort(c_thvalue, c_thvalue + end-start);


    return c_thvalue[(end-start)/2];
}

/*****************************************************
 * Exercise 4: partition                             *
 *****************************************************/
/**
 * Partitions the the array P wrt to its median value along a coordinate
 *
 * @param P a set of points (an array)
 * @param start the starting index
 * @param end the last index; the element P[end] is not considered
 * @param c the coordinate that we will consider the median
 * @return the index of the median value
 */

int partition(point* P, int start, int end, int c) {
    // The size of the sub-array
    int size = end - start;

    // Compute the index of the median value in the sub-array
    int medianIndex = start + (size / 2);

    // Partially sort the elements in the sub-array around the median using nth_element
    std::nth_element(P + start, P + medianIndex, P + end, [c](const point& a, const point& b) {
        return a[c] < b[c];
    });

    // Now, P[medianIndex] is the element that would be in that position if the array were sorted.
    // All elements before the median are less than or equal to it, all elements after are greater.

    return medianIndex;
}


/*****************************************************
 * Exercise 5: create_node                           *
 *****************************************************/
/**
 * Creates a leaf node in the kd-tree
 *
 * @param val the value of the leaf node
 * @return a leaf node that contains val
 */
// Function to create a leaf node
node* create_node(int _idx) {
    node* newNode = new node;
    newNode->idx = _idx; // Store the index of the data point
    newNode->c = -1; // -1 indicates that this is a leaf node, no split coordinate
    newNode->m = -1; // -1 indicates that this is a leaf node, no median value
    newNode->left = NULL; // No children for a leaf node
    newNode->right = NULL; // No children for a leaf node
    return newNode;
}

// Function to create an internal node
node* create_node(int _c, double _m, int _idx, node* _left, node* _right) {
    node* newNode = new node;
    newNode->c = _c; // Store the coordinate for the split
    newNode->m = _m; // Store the split value
    newNode->idx = _idx; // Store the index of the data point that represents the median value
    newNode->left = _left; // Assign the left child
    newNode->right = _right; // Assign the right child
    return newNode;
}


node* build(point* P, int start, int end, int c, int dim) {
    // builds tree for sub-cloud P[start -> end-1]
    assert(end - start >= 0);
    if (debug)
        std::cout << "start=" << start << ", end=" << end << ", c=" << c
                  << std::endl;
    if (end - start == 0)  // no data point left to process
        return NULL;
    else if (end - start == 1)  // leaf node
        return create_node(start);
    else {  // internal node
        if (debug) {
            std::cout << "array:\n";
            for (int i = start; i < end; i++)
                print_point(P[i], dim);
            // std::cout << P[i] << ((i==end-1)?"\n":" ");
        }
        // compute partition
        // rearrange subarray (less-than-median first, more-than-median last)
        int p = partition(P, start, end, c);
        if (p == -1) { return NULL; }
        double m = P[p][c];
        // prepare for recursive calls
        int cc = (c + 1) % dim;  // next coordinate
        return create_node(c, m, p, build(P, start, p, cc, dim),
                           build(P, p + 1, end, cc, dim));
    }
}

/*****************************************************
 * Exercise 6: defeatist_search                      *
 *****************************************************/
/**
 *  Defeatist search in a kd-tree
 *
 * @param n the roots of the kd-tree
 * @param q the query point
 * @param dim the dimension of the points
 * @param P a pointer to an array of points
 * @param res the distance of q to its NN in P
 * @param nnp the index of the NN of q in P
 */
void defeatist_search(node* n, point q, int dim, point* P, double& res, int& nnp) {
    if (n != NULL) {
        double current_dist = dist(q, P[n->idx], dim);
        if (current_dist < res) {
            res = current_dist; // Update the closest distance
            nnp = n->idx;       // Update the index of the nearest point
        }
        // Decide whether to go left or right in the tree
        if ((n->left != NULL || n->right != NULL) && (q[n->c] <= n->m)) {
            defeatist_search(n->left, q, dim, P, res, nnp);
        } else if (n->right != NULL) {
            defeatist_search(n->right, q, dim, P, res, nnp);
        }
    }
}


/*****************************************************
 * Exercise 7: backtracking_search                   *
 *****************************************************/
/**
 *  Backtracking search in a kd-tree
 *
 * @param n the roots of the kd-tree
 * @param q the query point
 * @param dim the dimension of the points
 * @param P a pointer to an array of points
 * @param res the distance of q to its NN in P
 * @param nnp the index of the NN of q in P
 */
void backtracking_search(node* n, point q, int dim, point* P, double& res, int& nnp) {
    if (n != NULL) {
        double current_dist = dist(q, P[n->idx], dim);
        if (current_dist < res) {
            res = current_dist; // Update the closest distance
            nnp = n->idx;      // Update the index of the nearest point
        }
        
        node* first_search = (q[n->c] <= n->m) ? n->left : n->right;
        node* second_search = (q[n->c] <= n->m) ? n->right : n->left;

        if (first_search != NULL) {
            backtracking_search(first_search, q, dim, P, res, nnp);
        }
        
        // Only search the other side if there's a chance of finding a closer point
        if (second_search != NULL && ((q[n->c] <= n->m && q[n->c] + res > n->m) || (q[n->c] > n->m && q[n->c] - res <= n->m))) {
            backtracking_search(second_search, q, dim, P, res, nnp);
        }
    }
}


