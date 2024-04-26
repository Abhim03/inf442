#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <ANN/ANN.h>
#include "Dataset.hpp"
#include "KnnRegression.hpp"

/** @file
 * Test suite for the impact of K in KNN Regression.
 *  This executable will put the provided CSV files in objects of class dataset 
 *  and perform KNN Regression for k=1 to 100
 *  for the samples in the test file using instances
 *  of the train file. It then prints the resulting MSE for each value of k.
*/
int main(int argc, char* argv[]) {
	if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <train_file> <test_file> [ <column_for_Regression> ]" << std::endl;
        return 1;
    }

	Dataset train_dataset(argv[1]);
	Dataset regr_dataset(argv[2]);
    
    int col_regr;
    if (argc == 4) {
    	col_regr = atoi(argv[3]);
    } else {
		col_regr = train_dataset.get_dim()-1;
		std::cout << "No column specified for Regression, assuming last column of dataset (" << col_regr << ")." << std::endl;    
    }

	train_dataset.show(false);

	assert(train_dataset.get_dim() == regr_dataset.get_dim());
    // otherwise doesn't make sense

	int dim = train_dataset.get_dim();

	FILE* fp;
	
	fp = fopen("output.txt", "w");

	fclose(fp);
	annClose();
	return 0;
}
