#pragma once

/** 
 * This function computes the mean of the given array of values 
 * 
 * @param values the array with the values
 * @param length the length of the array values
 * @return the mean of the values in the array
 */
double compute_mean (double values[], int length);

/** 
 * This function computes the variance of the given array of values 
 * 
 * @param values the array with the values
 * @param length the length of the array values
 * @return the variance of the values in the array
 */
double compute_variance (double values[], int length);

/** 
 * This function computes the unbiased sample variance of the given
 * array of values 
 * 
 * @param values the array with the values
 * @param length the length of the array values
 * @return the variance of the values in the array
 */
double compute_sample_variance (double values[], int length);

/** 
 * This function computes the standard deviation of the given
 * array of values 
 * 
 * @param values the array with the values
 * @param length the length of the array values
 * @return the variance of the values in the array
 */
double compute_standard_deviation (double values[], int length);

/** 
 * This function computes the unbiased sample standard deviation
 * of the given array of values 
 * 
 * @param values the array with the values
 * @param length the length of the array values
 * @return the variance of the values in the array
 */
double compute_sample_standard_deviation (double values[], int length);

/**
 * This function prints a rectangular matrix on the standard output, 
 * placing each row on a separate line.  
 * 
 * @param matrix the matrix to print
 * @param rows the number of rows in the matrix
 * @param columns the number of columns
 */
void print_matrix (double** matrix, int rows, int columns);

/** 
 * This function extracts one row from a data matrix
 * 
 * @param matrix the matrix with the data
 * @param columns the number of columns in the matrix
 * @param index the index of the row to extract
 * @param row the array where the extracted values are to be placed
 */
void get_row (double** matrix, int columns, int index, double row[]);

/** 
 * This function extracts one column from a data matrix
 * 
 * @param matrix the matrix with the data
 * @param rows the number of rows in the matrix
 * @param index the index of the column to extract
 * @param column the array where the extracted values are to be placed
 */
void get_column (double** matrix, int rows, int index, double column[]);

/**
 * This function computes the covariance of two vectors of data of the same length
 * @param values1 the first vector
 * @param values2 the second vector
 * @param length the length of the two vectors
 * @return the covariance of the two vectors
 */
double compute_covariance (double values1[], double values2[], int length);

/**
 * This function computes the correlation of two vectors of data of the same length
 * 
 * @param values1 the first vector
 * @param values2 the second vector
 * @param length the length of the two vectors
 * @return the correlation of the two vectors
 */
double compute_correlation (double values1[], double values2[], int length);

/**
 * This function computes the covariance matrix of the matrix provided as argument
 * 
 * @param matrix the input matrix 
 * @param rows the number of rows in the matrix
 * @param columns the number of columns in the matrix
 * @return the covariance matrix
 */
double** compute_covariance_matrix (double** data_matrix, int rows, int columns);

/**
 * This function computes the correlation matrix of the matrix provided as argument
 * 
 * @param matrix the input matrix 
 * @param rows the number of rows in the matrix
 * @param columns the number of columns in the matrix
 * @return the correlation matrix
 */
double** compute_correlation_matrix (double** data_matrix, int rows, int columns);

/************* Helper functions **************/

// Read the data matrix from the standard input
void read_matrix (double** matrix, int rows, int columns);

// Print an array on the standard output
void print_array (double values[], int length);

// Prepare an empty matrix
double** prepare_matrix(int rows, int columns);
