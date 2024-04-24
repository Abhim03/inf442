#include "Dataset.hpp"
#include <ANN/ANN.h>

#ifndef CLASSIFICATION_HPP
#define CLASSIFICATION_HPP

// The Classification class is an abstract class that will be the basis of the KnnClassification classe.
class Classification {
  protected:
    // The pointer to a dataset.
    Dataset *m_dataset;

    // The column to do classification on.
    int m_col_class;

  public:
    //The constructor sets private attributes dataset (as a pointer) and the column to do classification on (as an int).
    Classification(Dataset *dataset, int col_class);

    // The dataset getter.
    Dataset *get_dataset();
    
    // The col_class getter.
    int get_col_class() const;
    
    // The estimate method is virtual: it depends on the Classification models implemented (here we use only the KnnClassification class).
    virtual int estimate(const ANNpoint &y, double threshold = 0.5) const = 0;
};

#endif //CLASSIFICATION_HPP
