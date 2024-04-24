#include "Classification.hpp"
#include "Dataset.hpp"

Classification::Classification(Dataset* dataset, int col_class) {
    m_dataset = dataset;
    m_col_class = col_class;
}

Dataset* Classification::get_dataset(){
    return m_dataset;
}

int Classification::get_col_class() const {
    return m_col_class;
}
