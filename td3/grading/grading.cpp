#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstdarg>
#include <iterator>
#include <string>
#include <regex>
#include <numeric>
#include <cmath>
#include <fstream>
#include <random>
#include <limits>

#include "../gradinglib/gradinglib.hpp"
#include "../kmeans.cpp"


namespace tdgrading {

using namespace testlib;
using namespace std;

const double deps = 0.001;
const std::string default_path = "./grading/tests/";    

int test_point(std::ostream &out, const std::string test_name)
{
	start_test_suite(out, test_name);
	// tolerance for comparison of doubles
	const double eps = 0.0001;

	// dimension used for tests
	point::d = 2;

	point p;
	point q;

	std::vector<int> res;

	res.push_back( test_eq_approx(out, "point constructor", p.coords[0], 0.0, eps) );
	res.push_back( test_eq_approx(out, "point constructor", p.coords[1], 0.0, eps) );
	res.push_back( test_eq_approx(out, "squared_dist", p.squared_dist(q), 0.0, eps) );

	q.coords[0] = -1.0;
	q.coords[1] = 1.0;

	res.push_back( test_eq_approx(out, "squared_dist", q.squared_dist(q), 0.0, eps) );
	res.push_back( test_eq_approx(out, "squared_dist", p.squared_dist(q), 2.0, eps) );
	res.push_back( test_eq_approx(out, "squared_dist", q.squared_dist(p), 2.0, eps) );

	return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}

int test_intracluster_variance(std::ostream &out, const std::string test_name)
{
	start_test_suite(out, test_name);

	std::vector<int> res;

	// tolerance for comparison of doubles
	const double eps = 0.0001;

	// dimension used for tests
	point::d = 1;

	// temporary container
	point p;

	// test case 1
	const double dist_onepoint_zerodist = 0.0;
	cloud onepoint_zerodist(1, 1, 1);
	p.coords[0] = 0.0;
	onepoint_zerodist.add_point(p, 0);
	res.push_back( test_eq_approx(out, "intracluster_variance", onepoint_zerodist.intracluster_variance(), dist_onepoint_zerodist, eps) );

	// test case 2
	const double dist_onepoint_posdist = 0.25;
	cloud onepoint_posdist(1, 1, 1);
	p.coords[0] = 0.5;
	onepoint_posdist.add_point(p, 0);
	res.push_back( test_eq_approx(out, "intracluster_variance", onepoint_posdist.intracluster_variance(), dist_onepoint_posdist, eps) );

	// test case 3
	const double dist_twopoints = 0.625;
	cloud twopoints(1, 2, 1);
	p.coords[0] = -1.0;
	twopoints.add_point(p, 0);
	p.coords[0] = 0.5;
	twopoints.add_point(p, 0);
	p.coords[0] = -0.5;
	twopoints.set_center(p, 0);
	res.push_back( test_eq_approx(out, "intracluster_variance", twopoints.intracluster_variance(), dist_twopoints, eps) );

	// test case 4
	const double dist_twoclusters = 6.8125;
	cloud twoclusters(1, 4, 2);
	p.coords[0] = -1.0;
	twoclusters.add_point(p, 0);
	p.coords[0] = 0.5;
	twoclusters.add_point(p, 0);
	p.coords[0] = -0.5;
	twoclusters.set_center(p, 0);
	p.coords[0] = -2.0;
	twoclusters.add_point(p, 1);
	p.coords[0] = 2.0;
	twoclusters.add_point(p, 1);
	p.coords[0] = -3.0;
	twoclusters.set_center(p, 1);
	res.push_back( test_eq_approx(out, "intracluster_variance", twoclusters.intracluster_variance(), dist_twoclusters, eps) );

	return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}

int test_voronoi(std::ostream &out, const std::string test_name)
{
	start_test_suite(out, test_name);
	std::vector<int> res;


	{
	        // test case 1: d=1, three points, two centers
	        cloud c(1, 3, 2);
        	point p;
	
	        p.coords[0] = -3.0;
	        c.add_point(p, 1);
	
	        p.coords[0] = -1.0;
	        c.add_point(p, 1);
	
	        p.coords[0] = 0.0;
	        c.add_point(p, 1);
	
	        p.coords[0] = -2.0;
	        c.set_center(p, 0);
	
	        p.coords[0] = 1.9;
	        c.set_center(p, 1);
	
	        int nb = c.set_voronoi_labels();
	
		res.push_back( test_eq(out, "get_label", c.get_point(0).label, 0) );
		res.push_back( test_eq(out, "get_label", c.get_point(1).label, 0) );
		res.push_back( test_eq(out, "get_label", c.get_point(2).label, 1) );
		res.push_back( test_eq(out, "return", nb, 2) );
	}

	{
	        // test case 2: d=2, three points, two centers
	        cloud c(2, 3, 2);
		point p;
	
	        p.coords[0] = -3.0;
	        p.coords[1] = 3.0;
	        c.add_point(p, 1);
	
	        p.coords[0] = -1.0;
	        p.coords[1] = 0.0;
	        c.add_point(p, 1);
	
	        p.coords[0] = 2.0;
	        p.coords[1] = 1.0;
	        c.add_point(p, 1);
	
	        p.coords[0] = -5.0;
	        p.coords[1] = -2.0;
	        c.set_center(p, 0);
	
	        p.coords[0] = 0.0;
	        p.coords[1] = -2.0;
	        c.set_center(p, 1);
	
	        int nb = c.set_voronoi_labels();
	
		res.push_back( test_eq(out, "get_label", c.get_point(0).label, 0) );
		res.push_back( test_eq(out, "get_label", c.get_point(1).label, 1) );
		res.push_back( test_eq(out, "get_label", c.get_point(2).label, 1) );
		res.push_back( test_eq(out, "return", nb, 1) );
	}

	return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}

int test_centroids(std::ostream &out, const std::string test_name)
{
	start_test_suite(out, test_name);

	std::vector<int> res;

	// tolerance for comparison of doubles
        const double eps = 0.0001;

	{
	        // d=1, three points, three centers, no Voronoi
	        cloud c(1, 3, 3);
	        point p;
	        p.coords[0] = -3.0;
	        c.add_point(p, 0);
	        c.set_center(p, 0);
	        p.coords[0] = -1.0;
	        c.add_point(p, 1);
	        c.set_center(p, 1);
	        p.coords[0] = -2.1;
	        c.add_point(p, 1);
		p.coords[0] = 7.6;
		c.set_center(p, 2);
	
	        c.set_centroid_centers();
	
		res.push_back( test_eq_approx(out, "get_center", c.get_center(0).coords[0], -3.0, eps) );
		res.push_back( test_eq_approx(out, "get_center", c.get_center(1).coords[0], -1.55, eps) );
		res.push_back( test_eq_approx(out, "get_center", c.get_center(2).coords[0], 7.6, eps) );
	}

	{
        	// d=1, three points, two centers, with Voronoi
	        cloud c(1, 3, 2);
		point p;
	        p.coords[0] = -3.0;
	        c.add_point(p, 0);
	        c.set_center(p, 0);
	        p.coords[0] = -1.0;
	        c.add_point(p, 1);
	        c.set_center(p, 1);
	        p.coords[0] = -2.1;
	        c.add_point(p, 1);
	
	        c.set_voronoi_labels();
	        c.set_centroid_centers();
	
		res.push_back( test_eq_approx(out, "get_center", c.get_center(0).coords[0], -2.55, eps) );
		res.push_back( test_eq_approx(out, "get_center", c.get_center(1).coords[0], -1.0, eps) );
	}

	{
	        // d=2, three points, two centers, with Voronoi
	        cloud c(2, 3, 2);
		point p;
	        p.coords[0] = -3.0;
	        p.coords[1] = 0.0;
	        c.add_point(p, 0);
	        c.set_center(p, 0);
	        p.coords[0] = -1.0;
	        p.coords[1] = 0.0;
	        c.add_point(p, 1);
	        c.set_center(p, 1);
	        p.coords[0] = -2.1;
	        p.coords[1] = 0.0;
	        c.add_point(p, 1);
	
	        c.set_voronoi_labels();
	        c.set_centroid_centers();
	
		res.push_back( test_eq_approx(out, "get_center", c.get_center(0).coords[0], -2.55, eps) );
		res.push_back( test_eq_approx(out, "get_center", c.get_center(1).coords[0], -1.0, eps) );
	}

	return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}

int test_lloyd(std::ostream &out, const std::string test_name)
{
	start_test_suite(out, test_name);

	std::vector<int> res;

	// tolerance for comparison of doubles
        const double eps = 0.0001;
        cloud c1(2, 4, 2);

        // temporary container
        point p;

        // test case 1: d=1, three points, two centers, no Voronoi
        p.coords[0] = 1.0;
        p.coords[1] = 1.0;
        c1.add_point(p, 0);
        p.coords[0] = 2.0;
        p.coords[1] = 1.0;
        c1.add_point(p, 1);
        p.coords[0] = 4.0;
        p.coords[1] = 3.0;
        c1.add_point(p, 1);
        p.coords[0] = 5.0;
        p.coords[1] = 4.0;
        c1.add_point(p, 1);

        p.coords[0] = 1.0;
        p.coords[1] = 1.0;
        c1.set_center(p, 0);
        p.coords[0] = 2.0;
        p.coords[1] = 1.0;
        c1.set_center(p, 1);

        c1.set_centroid_centers();
	c1.lloyd();

	if(std::abs(c1.get_center(0).coords[0] - 1.5) <= eps)
	{
		res.push_back( test_eq_approx(out, "get_center", c1.get_center(0).coords[0], 1.5, eps) );
		res.push_back( test_eq_approx(out, "get_center", c1.get_center(1).coords[0], 4.5, eps) );
		res.push_back( test_eq(out, "get_label", c1.get_point(0).label, 0) );
		res.push_back( test_eq(out, "get_label", c1.get_point(1).label, 0) );
		res.push_back( test_eq(out, "get_label", c1.get_point(2).label, 1) );
	} else {
		res.push_back( test_eq_approx(out, "get_center", c1.get_center(0).coords[0], 4.5, eps) );
		res.push_back( test_eq_approx(out, "get_center", c1.get_center(1).coords[0], 1.5, eps) );
		res.push_back( test_eq(out, "get_label", c1.get_point(0).label, 1) );
		res.push_back( test_eq(out, "get_label", c1.get_point(1).label, 1) );
		res.push_back( test_eq(out, "get_label", c1.get_point(2).label, 0) );
	}

	return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}

int test_init_forgy(std::ostream &out, const std::string test_name)
{
	start_test_suite(out, test_name);

	std::vector<int> res;

	// number of random experiments
	const int K = 10000;
	// tolerance in probability
	const double delta = 0.0625;

	// dimenstion used for tests
	point::d = 1;

	// temporary container
	point p;

	const double prob_threepoints = 0.3333;
	cloud threepoints(1, 3, 2);
	p.coords[0] = 0.0;
	threepoints.add_point(p, 0);
	p.coords[0] = 1.0;
	threepoints.add_point(p, 0);
	p.coords[0] = 2.0;
	threepoints.add_point(p, 0);
	int cnt = 0;
	bool uniqueness = true;
	for(int k = 0; k < K; k++)
	{
		threepoints.init_forgy();
		if(threepoints.get_center(0).coords[0] == 1.0)
			cnt++;
		if(threepoints.get_center(0).coords[0] == threepoints.get_center(1).coords[0])
			uniqueness = false;
	}
	res.push_back( test_eq_approx(out, "init_forgy", cnt/(double)K, prob_threepoints, delta) );
	res.push_back( test_eq(out, "uniqueness", uniqueness, true) );

	return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}

int test_init_plusplus(std::ostream &out, const std::string test_name)
{
	start_test_suite(out, test_name);

	std::vector<int> res;

	// number of random experiments
	const int K = 10000;
	// tolerance in probability
	const double delta = 0.0625;

	// dimenstion used for tests
	point::d = 1;

	// temporary container
	point p;

	// test case 1
	const double prob_threepoints = 0.3333;
	cloud threepoints(1, 3, 1);
	p.coords[0] = 0.0;
	threepoints.add_point(p, 0);
	p.coords[0] = 1.0;
	threepoints.add_point(p, 0);
	p.coords[0] = 2.0;
	threepoints.add_point(p, 0);
	int cnt = 0;
	for(int k = 0; k < K; k++)
	{
		threepoints.init_plusplus();
		if(threepoints.get_center(0).coords[0] == 1.0)
			cnt++;
	}
	res.push_back( test_eq_approx(out, "init_plusplus", cnt/(double)K, prob_threepoints, delta) );

	// test case 2
	const double prob_twoclusters = 0.125;
	cloud twoclusters(1, 4, 2);
	p.coords[0] = 0.0;
	twoclusters.add_point(p, 0);
	p.coords[0] = 0.0;
	twoclusters.add_point(p, 0);
	p.coords[0] = 1.0;
	twoclusters.add_point(p, 0);
	p.coords[0] = 2.0;
	twoclusters.add_point(p, 0);
	cnt = 0;
	for(int k = 0; k < K; k++)
	{
		twoclusters.init_plusplus();
		if(twoclusters.get_center(1).coords[0] == 1.0)
			cnt++;
	}
	res.push_back( test_eq_approx(out, "init_plusplus", cnt/(double)K, prob_twoclusters, delta) );

	return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}

int test_init_random_partition(std::ostream &out, const std::string test_name)
{
	start_test_suite(out, test_name);

	std::vector<int> res;

	// number of random experiments
	const int K = 10000;
	// tolerance in probability
	const double delta = 0.0625;

	// dimenstion used for tests
	point::d = 1;

	// temporary container
	point p;

	const double prob_threepoints = 0.3333;
	cloud threepoints(1, 3, 3);
	p.coords[0] = 0.0;
	threepoints.add_point(p, 0);
	p.coords[0] = 1.0;
	threepoints.add_point(p, 0);
	p.coords[0] = 2.0;
	threepoints.add_point(p, 0);
	int cnt = 0;
	for(int k = 0; k < K; k++)
	{
		threepoints.init_random_partition();
		if(threepoints.get_point(2).label == 1)
			cnt++;
	}
	res.push_back( test_eq_approx(out, "init_random_partition", cnt/(double)K, prob_threepoints, delta) );

	return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}

//-------------------------------------------------------------------



int grading(std::ostream &out, const int test_case_number)
{
/**

Annotations used for the autograder.

[START-AUTOGRADER-ANNOTATION]
{
  "total" : 8,
  "names" : [
      "kmeans.cpp::test_point", 
      "kmeans.cpp::test_intracluster_variance", 
      "kmeans.cpp::test_voronoi", 
      "kmeans.cpp::test_centroids", 
      "kmeans.cpp::test_init_random_partition",
      "kmeans.cpp::test_lloyd", 
      "kmeans.cpp::test_init_forgy",
      "kmeans.cpp::test_init_plusplus"
        ],
  "points" : [10, 10, 10, 10, 10, 10, 10, 10]
}
[END-AUTOGRADER-ANNOTATION]
*/

    int const total_test_cases = 8;
    std::string const test_names[total_test_cases] = {"test_point", "test_intracluster_variance", "test_voronoi", "test_centroids", "test_init_random_partition", "test_lloyd", "test_init_forgy", "test_init_plusplus"};
    int const points[total_test_cases] = {10, 10, 10, 10, 10, 10, 10, 10};
    int (*test_functions[total_test_cases]) (std::ostream &, const std::string) = {
      test_point, test_intracluster_variance, test_voronoi, test_centroids, test_init_random_partition, test_lloyd, test_init_forgy, test_init_plusplus
    };

    return run_grading(out, test_case_number, total_test_cases,
                       test_names, points,
                       test_functions);
}

} // End of namepsace tdgrading
