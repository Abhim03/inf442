// Include the library that defines input/output streams and operations on them
#include <iostream>
// Include the library with basic mathematical functions, e.g. sqrt
#include <cmath>
// Need that for memset
#include <cstring>

/********************
 * Helper functions *
 ********************/

// Read the data matrix from the standard input
void read_matrix (double** matrix, int rows, int columns) {
    for (int i = 0; i < rows; ++i) {
        matrix[i] = new double[columns];
        for (int j = 0; j < columns; ++j)
            std::cin >> matrix[i][j];
    }
}

// Print an array on the standard output
void print_array (double values[], int length) {
    for (int i = 0; i < length - 1; i++)
        std::cout << values[i] << " ";

    std::cout << values[length - 1];
}

// Prepare an empty matrix
double** prepare_matrix(int rows, int columns) {
    double** matrix = new double* [rows];
    for (int i = 0; i < rows; ++i) {
        matrix[i] = new double[columns];
        for (int j = 0; j < columns; ++j)
            matrix[i][j] = 0;
    }

    return matrix;
}

/*****************************************************
 * Exercise 1: Mean, variance and standard deviation *
 *****************************************************/

/** 
 * This function computes the mean of the given array of values 
 * 
 * @param values the array with the values
 * @param length the length of the array 'values'
 * @return the mean of the values in the array
 */
double compute_mean (double values[], int length) {
    double sum = 0;

    // TODO
    for(int i=0;i<length;i++){
        sum += values[i];
    }

    return sum / length;
}

/** 
 * This function computes the variance of the given array of values 
 * 
 * @param values the array with the values
 * @param length the length of the array 'values'
 * @return the variance of the values in the array
 */
double compute_variance (double values[], int length) {
    double mean = compute_mean(values, length);
    double sum = 0;
    // TODO
    for(int i=0;i<length;i++){
        sum += (values[i]-mean)*(values[i]-mean);
    }
    // Do not forget to replace this return by a correct one!
    return sum/length;
}


/** 
 * This function computes the unbiased sample variance of the given
 * array of values 
 * 
 * @param values the array with the values
 * @param length the length of the array values
 * @return the variance of the values in the array
 */
double compute_sample_variance (double values[], int length) {
    // TODO
    // Note that you can call the functions defined above
    double variance = compute_variance(values,length);

    // Do not forget to replace this return by a correct one!
    return (double)length/((double)length-1.0)*variance;
}

/** 
 * This function computes the standard deviation of the given
 * array of values 
 * 
 * @param values the array with the values
 * @param length the length of the array values
 * @return the variance of the values in the array
 */
double compute_standard_deviation (double values[], int length) {
    // TODO
    // Note that you can call the functions defined above

    // Do not forget to replace this return by a correct one!
    return std::sqrt(compute_variance(values,length));
}

/** 
 * This function computes the unbiased sample standard deviation
 * of the given array of values 
 * 
 * @param values the array with the values
 * @param length the length of the array values
 * @return the variance of the values in the array
 */
double compute_sample_standard_deviation (double values[], int length) {
    // TODO
    // Note that you can call the functions defined above

    // Do not forget to replace this return by a correct one!
    return std::sqrt(compute_sample_variance(values,length));
}

/*************************************
 * Exercise 2: Working with matrices *
 *************************************/

/**
 * This function prints a rectangular matrix on the standard output, 
 * placing each row on a separate line.  
 * 
 * @param matrix the matrix to print
 * @param rows the number of rows in the matrix
 * @param columns the number of columns
 */
void print_matrix (double** matrix, int rows, int columns) {
    // TODO

    for(int i=0;i<rows;i++){
        for(int j=0;j<columns-1;j++){
            std::cout << matrix[i][j] << " ";
        }
        std::cout << matrix[i][columns-1] << std::endl;
    }
}

/** 
 * This function extracts one row from a data matrix
 * 
 * @param matrix the matrix with the data
 * @param columns the number of columns in the matrix
 * @param index the index of the row to extract
 * @param row the array where the extracted values are to be placed
 */
void get_row (double** matrix, int columns, int index, double row[]) {
    // TODO
    for(int i=0;i<columns;i++){
        row[i] = matrix[index][i];
    }
}

/** 
 * This function extracts one column from a data matrix
 * 
 * @param matrix the matrix with the data
 * @param rows the number of rows in the matrix
 * @param index the index of the column to extract
 * @param column the array where the extracted values are to be placed
 */
void get_column (double** matrix, int rows, int index, double column[]) {
    // TODO
    for(int i=0;i<rows;i++){
        column[i] = matrix[i][index];
    }
}

/****************************************************
 * Exercise 3: Covariance and correlation of arrays *
 ****************************************************/

/**
 * This function computes the covariance of two vectors of data of the same length
 * @param values1 the first vector
 * @param values2 the second vector
 * @param length the length of the two vectors
 * @return the covariance of the two vectors
 */
double compute_covariance(double values1[], double values2[], int length) {
    // TODO
    // Do not forget to replace this return by a correct one!
    double cov = 0;
    double mean1 = compute_mean(values1,length);
    double mean2 = compute_mean(values2,length);
    for(int i=0;i<length;i++){
        cov += (values1[i]-mean1)*(values2[i]-mean2);
    }
    return cov/length;
}

/**
 * This function computes the correlation of two vectors of data of the same length
 * 
 * @param values1 the first vector
 * @param values2 the second vector
 * @param length the length of the two vectors
 * @return the correlation of the two vectors
 */
double compute_correlation(double values1[], double values2[], int length) {
    // TODO: Use computeCovariance() and functions already implemented
    // Do not forget to replace this return by a correct one!
    double cov = compute_covariance(values1,values2,length);
    double dev1 = compute_standard_deviation(values1,length);
    double dev2 = compute_standard_deviation(values2,length);
    return cov/dev1/dev2;
}

/***************************************************
 * Exercise 4: Covariance and correlation matrices *
 ***************************************************/

/**
 * This function computes the covariance matrix of the matrix provided as argument
 * 
 * @param data_matrix the input matrix 
 * @param rows the number of rows in the matrix
 * @param columns the number of columns in the matrix
 * @return the covariance matrix
 */
double** compute_covariance_matrix (double** data_matrix, int rows, int columns) {
    // Initialise a square matrix
    double** matrix = prepare_matrix(columns, columns);
    // Prepare temporary storage for columns
    double column1[rows], column2[rows];

    //std::cout << rows << " : " << columns << std::endl;

    for(int i=0;i<columns;i++){
        get_column(data_matrix, rows, i, column1);
        for(int j=0;j<columns;j++){
            get_column(data_matrix, rows, j, column2);
            matrix[i][j] = compute_covariance(column1,column2,rows);
        }
    }

    // TODO
    return matrix;
}

/**
 * This function computes the correlation matrix of the matrix provided as argument
 * 
 * @param data_matrix the input matrix 
 * @param rows the number of rows in the matrix
 * @param columns the number of columns in the matrix
 * @return the correlation matrix
 */
double** compute_correlation_matrix (double** data_matrix, int rows, int columns) {
    // Initialise a square matrix
    double** matrix = prepare_matrix(columns, columns);
    // Prepare temporary storage for columns
    double column1[rows], column2[rows];

    for(int i=0;i<columns;i++){
        get_column(data_matrix, rows, i, column1);
        for(int j=0;j<columns;j++){
            get_column(data_matrix, rows, j, column2);
            matrix[i][j] = compute_correlation(column1,column2,rows);
        }
    }

    // TODO
    return matrix;
}
