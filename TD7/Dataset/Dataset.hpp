#ifndef DATASET_HPP
#define DATASET_HPP

#include <iostream>
#include <vector>
#include <cstdlib>
#include <string>

/*! \mainpage Documentation of TD7
 *
 * Welcome to the documentation of INF442's TD number 7.
 * To explore the classes you have to implement in this TD, go to Classes > Class list.
 * To get some pointers on how to implement the test_* executables, go to Files > Files list.
*/

/**
  The dataset class encapsulates a dataset in a vector of vectors and provides a kind of interface to manipulate it.
*/
class Dataset {
    public:
      /**
        The constructor needs the path of the file as a string.
      */
      Dataset(const char* file);
        
      /**
        Standard destructor
      */
      ~Dataset();
        
      /**
        The show method displays the number of instances and columns of the dataset.
        @param verbose If set to True, the dataset is also printed.
      */
      void show(bool verbose) const;
        
      /**
        Returns a copy of an instance.
        @param i Instance number (= row) to get.
      */
    	const std::vector<double>& get_instance(int i) const;
      
      /**
          The getter to the number of instances / samples.
        */
    	int get_nbr_samples() const;
        
      /**
          The getter to the dimension of the dataset.
      */
    	int get_dim() const;

    private:
        /**
          The dimension of the dataset.
        */
		int m_dim;
        /**
          The number of instances / samples.
        */
		int m_nsamples;
        /**
          The dataset is stored as a vector of double vectors.
        */
        std::vector<std::vector<double> > m_instances;
};
#endif //DATASET_HPP
