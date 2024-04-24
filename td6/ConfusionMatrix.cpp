#include "ConfusionMatrix.hpp"
#include <iostream>

using namespace std;

ConfusionMatrix::ConfusionMatrix() {
    // Initialize all elements of the confusion matrix to zero
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            m_confusion_matrix[i][j] = 0;}}}

ConfusionMatrix::~ConfusionMatrix() {
    // No dynamic memory, empty destructor
}

void ConfusionMatrix::add_prediction(int true_label, int predicted_label) {
    // Increment the appropriate cell in the confusion matrix
    if (true_label >= 0 && true_label < 2 && predicted_label >= 0 && predicted_label < 2) {
        m_confusion_matrix[true_label][predicted_label]++;
    }
}

void ConfusionMatrix::print_evaluation() const {
    cout << "\t\tPredicted\n";
    cout << "\t\t0\t1\n";
    cout << "Actual\t0\t" << get_tn() << "\t" << get_fp() << endl;
    cout << "\t1\t" << get_fn() << "\t" << get_tp() << endl << endl;

    cout << "Error rate:\t\t" << error_rate() << endl;
    cout << "False alarm rate:\t" << false_alarm_rate() << endl;
    cout << "Detection rate:\t\t" << detection_rate() << endl;
    cout << "F-score:\t\t" << f_score() << endl;
    cout << "Precision:\t\t" << precision() << endl;
}

int ConfusionMatrix::get_tp() const {
    return m_confusion_matrix[1][1];
}

int ConfusionMatrix::get_tn() const {
    return m_confusion_matrix[0][0];
}

int ConfusionMatrix::get_fp() const {
    return m_confusion_matrix[0][1];
}

int ConfusionMatrix::get_fn() const {
    return m_confusion_matrix[1][0];
}

double ConfusionMatrix::f_score() const {
    double prec = precision();
    double rec = detection_rate();
    return (prec + rec == 0) ? 0 : (2 * prec * rec) / (prec + rec);
}

double ConfusionMatrix::precision() const {
    int tp_plus_fp = get_tp() + get_fp();
    return tp_plus_fp == 0 ? 0 : static_cast<double>(get_tp()) / tp_plus_fp;
}

double ConfusionMatrix::error_rate() const {
    int total = get_tp() + get_tn() + get_fp() + get_fn();
    return total == 0 ? 0 : static_cast<double>(get_fp() + get_fn()) / total;
}

double ConfusionMatrix::detection_rate() const {
    int tp_plus_fn = get_tp() + get_fn();
    return tp_plus_fn == 0 ? 0 : static_cast<double>(get_tp()) / tp_plus_fn;
}

double ConfusionMatrix::false_alarm_rate() const 
{
    int fp_plus_tn = get_fp() + get_tn();
    return fp_plus_tn == 0 ? 0 : static_cast<double>(get_fp()) / fp_plus_tn;
}
