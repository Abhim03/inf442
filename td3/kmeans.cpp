#include <iostream>
#include <cassert>
#include <cmath>	// for sqrt, fabs
#include <cfloat>	// for DBL_MAX
#include <cstdlib>	// for rand, srand
#include <ctime>	// for rand seed
#include <fstream>
#include <cstdio>	// for EOF
#include <string>
#include <algorithm>	// for count
#include <vector>

using std::rand;
using std::srand;
using std::time;


class point
{
    public:

		// Default constructor
		point() : label(0), coords(new double[d]) {
			for (int i = 0; i < d; i++) {
				coords[i] = 0.0;
			}
		}

		// Copy constructor for deep copying
		point(const point& other) : label(other.label), coords(new double[d]) {
			for (int i = 0; i < d; i++) {
				coords[i] = other.coords[i];
			}
		}

		// Assignment operator for deep copying
		point& operator=(const point& other) {
			if (this != &other) { // Protect against self-assignment
				label = other.label;
				// Reallocate memory before copying the values
				delete[] coords;
				coords = new double[d];
				for (int i = 0; i < d; i++) {
					coords[i] = other.coords[i];
				}
			}
			return *this;
		}
		~point(){
			delete[] coords;
		}
		
		void print() const{
			for(int i=0;i<d-1;i++){
				std::cout << coords[i] << '\t';
			}
			if(d>0){
				std:: cout << coords[d-1];
			}
			std::cout << "\n";
		}

		double squared_dist(const point &q) const {
			double dist=0;
			for(int i=0;i<d;i++){
				dist+= pow(coords[i]-q.coords[i],2);
			}

			return dist;
		}

        static int d;
        double *coords;
        int label;
};

int point::d;

class cloud
{
	private:

	int d;
	int n;
	int k;

	// maximum possible number of points
	int nmax;

	point *points;
	point *centers;


	public:

	cloud(int _d, int _nmax, int _k)
	{
		d = _d;
		point::d = _d;
		n = 0;
		k = _k;

		nmax = _nmax;

		points = new point[nmax];
		centers = new point[k];

		srand(time(0));
	}

	~cloud()
	{
		delete[] centers;
		delete[] points;
	}

	void add_point(const point &p, int label)
	{
		for(int m = 0; m < d; m++)
		{
			points[n].coords[m] = p.coords[m];
		}

		points[n].label = label;

		n++;
	}

	int get_d() const
	{
		return d;
	}

	int get_n() const
	{
		return n;
	}

	int get_k() const
	{
		return k;
	}

	point &get_point(int i)
	{
		return points[i];
	}

	point &get_center(int j)
	{
		return centers[j];
	}

	void set_center(const point &p, int j)
	{
		for(int m = 0; m < d; m++)
			centers[j].coords[m] = p.coords[m];
	}

	double intracluster_variance() const {
    double totalVariance = 0.0;

    for (int i = 0; i < n; i++) {
        int centerIndex = points[i].label;
        if (centerIndex >= 0 && centerIndex < k) {
            double distSquared = points[i].squared_dist(centers[centerIndex]);
            totalVariance += distSquared;
        }
    }

    if (n > 0) {
        totalVariance /= n;
    }

    return totalVariance;
}

	int set_voronoi_labels() {
    int changes = 0; 

    
    for (int i = 0; i < n; i++) {
        double minDist = DBL_MAX; 
        int closestCenterIndex = -1; 

        
        for (int j = 0; j < k; j++) {
            double currentDist = points[i].squared_dist(centers[j]);

            
            if (currentDist < minDist) {
                minDist = currentDist;
                closestCenterIndex = j;
            }
        }

        
        if (points[i].label != closestCenterIndex) {
            points[i].label = closestCenterIndex;
            changes++;
        }
    }

    
    return changes;
}

		void set_centroid_centers() {
		std::vector<int> count(k, 0); // To count the number of points in each cluster

		// Initialize centroids to 0
		for (int j = 0; j < k; j++) {
			for (int m = 0; m < d; m++) {
				centers[j].coords[m] = 0;
			}
		}

		// Sum up all points coordinates in their respective cluster
		for (int i = 0; i < n; i++) {
			int cluster = points[i].label;
			for (int m = 0; m < d; m++) {
				centers[cluster].coords[m] += points[i].coords[m];
			}
			count[cluster]++;
		}

		// Divide by the number of points in each cluster to get the mean (centroid)
		for (int j = 0; j < k; j++) {
			if (count[j] > 0) { // Avoid division by zero if cluster is empty
				for (int m = 0; m < d; m++) {
					centers[j].coords[m] /= count[j];
				}
			}
		}
	}


	void init_random_partition() {
    for (int i = 0; i < n; i++) {
        points[i].label = rand() % k; // Assign to a random cluster
    }

    set_centroid_centers(); // Update centers to centroids of new random partitions
}


		void lloyd() {
		bool changed = true;

		while (changed) {
			int changes = set_voronoi_labels(); // Assign points to nearest center
			set_centroid_centers(); // Update centers to centroids
			
			changed = (changes > 0); // Continue if there were changes
		}
	}


	void init_forgy()
	{
	}

	void init_plusplus()
	{
	}
};
