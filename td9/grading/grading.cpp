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
#include "../node.hpp"
#include "../activation.hpp"
#include "../neuron.hpp"
#include "../perceptron.hpp"

namespace tdgrading {

using namespace testlib;
using namespace std;
using std::cout;
using std::endl;

class OLP_test : public OneLayerPerceptron
{
public:
    enum OLPERR
    {
        OK,
        NO,
        SIZE,
        INPUTS,
        ACT,
        ACTDER
    };
    static double eps;

    OLP_test(int _dim, int _size, double _rate, double _decay) : OneLayerPerceptron(_dim, _size, _rate, _decay, sigma, sigma_der) {}
    OLP_test(int _dim, int _size, double _rate, double _decay,
        std::function<double(double)> _activation, std::function<double(double)> _activation_der) :
        OneLayerPerceptron(_dim, _size, _rate, _decay, _activation, _activation_der) {}

    bool test_inputs();
    OLPERR test_hidden();
    OLPERR test_output();
    bool test_act(std::function<double(double)> a, std::function<double(double)> da);

    static bool interpret_neuron_test(OLPERR code, std::string type);

    bool test_prepareInputs(Dataset *data, int row, int regr);
    bool test_computeHiddenStep();
    bool test_computeOutputStep(Dataset *data, int row, int regr);
    bool test_propagateBackHidden();
};

double OLP_test::eps = 0.0001;

bool OLP_test::test_inputs()
{
    for (int i = 0; i < dim; i++)
        if (inputs[i] == nullptr)
            return false;

    return true;
}

OLP_test::OLPERR OLP_test::test_hidden()
{
    for (int i = 0; i < size; i++)
        if (hidden[i] == nullptr)
            return NO;

    for (int i = 0; i < size; i++)
        if (hidden[i]->get_nb_dendrites() != dim)
            return SIZE;

    for (int i = 0; i < size; i++)
        for (int j = 0; j < dim; j++)
            if (hidden[i]->get_dendrite(j) != inputs[j])
                return INPUTS;

    return OK;
}

OLP_test::OLPERR OLP_test::test_output()
{
    if (output == nullptr)
        return NO;

    if (output->get_nb_dendrites() != size)
        return SIZE;

    for (int i = 0; i < size; i++)
        if (output->get_dendrite(i) != hidden[i]->get_axon())
            return INPUTS;

    return OK;
}

bool OLP_test::interpret_neuron_test(OLP_test::OLPERR code, std::string type)
{
    switch (code)
    {
    case OLP_test::NO:
        cout << "[NOK] - " << type << " neuron is not initialised correctly" << endl;
        return false;

    case OLP_test::SIZE:
        cout << "[NOK] - " << type << " neuron has incorrect number of dendrites" << endl;
        return false;

    case OLP_test::INPUTS:
        cout << "[NOK] - " << type << " neuron is not correctly connected to inputs" << endl;
        return false;

    case OLP_test::OK:
        return true;

    default:
        cout << "[NOK] - " << type << " neuron has an unknown problem" << endl;
        return false;
    }
}

bool OLP_test::test_act(std::function<double(double)> a, std::function<double(double)> da)
{
    for (int i = 0; i < dim; i++)
    {
        inputs[i]->set_signal(1);
        for (int j = 0; j < size; j++)
            hidden[j]->set_weight(i, 1);
    }
    double ad = a(dim);
    for (int i = 0; i < size; i++)
    {
        output->set_weight(i, 1);
        hidden[i]->step();
        if (fabs(hidden[i]->get_axon()->get_signal() - ad) > eps)
        {
            cout << "[NOK] - Incorrect activation function used for a hidden neuron (did you hardcode sigma instead of using the argument?)" << endl;
            return false;
        }
    }
    output->step();
    if (fabs(output->get_axon()->get_signal() - ad * size) > eps)
    {
        cout << "[NOK] - Incorrect activation function used for the output neuron (did you hardcode sigma instead of using the argument?)" << endl;
        return false;
    }
    output->set_back_value(1);
    output->step_back();
    for (int i = 0; i < size; i++)
    {
        if (fabs(output->get_weight(i) + ad - 1) > eps)
        {
            cout << "[NOK] - Incorrect activation function derivative used for the output neuron (did you hardcode sigma_der instead of using the argument?)" << endl;
            return false;
        }
        hidden[i]->set_back_value(1);
        hidden[i]->step_back();
    }
    double dad = da(dim);
    for (int i = 0; i < size; i++)
        for (int j = 0; j < dim; j++)
            if (fabs(hidden[i]->get_weight(j) + dad - 1) > eps)
            {
                cout << "[NOK] - Incorrect activation function derivative used for a hidden neuron (did you hardcode sigma_der instead of using the argument?)" << endl;
                return false;
            }
    return true;
}

bool OLP_test::test_prepareInputs(Dataset *data, int row, int regr) {
    prepare_inputs(data, row, regr);

    std::vector<double> *mins = data->get_mins();
    std::vector<double> *maxs = data->get_maxs();

    for (int j = 0; j < size; j++)
        for (int i = 0; i < dim; i++)
            hidden[j]->set_weight(i, 1);

    if (regr > 0) {
        double value = data->get_instance(row)[regr-1];
        double norm = normalize(value, data, regr-1);

        if (fabs(inputs[regr-1]->get_signal() - value) < eps) {
            cout << "[NOK] - Input data not normalised" << endl;
            return false;
        }

        if (fabs(inputs[regr-1]->get_signal() - norm) > eps) {
            cout << "[NOK] - Input data not initialised correctly" << endl;
            return false;
        }
    }

    if (regr <= dim) {
        double value = data->get_instance(row)[regr+1];
        double shifted_value = data->get_instance(row)[regr];
        double norm = normalize(value, data, regr + 1);
        double shifted_norm = normalize(shifted_value, data, regr);

        if (fabs(inputs[regr]->get_signal() - value) < eps) {
            cout << "[NOK] - Input data not normalised" << endl;
            return false;
        }

        if (fabs(inputs[regr]->get_signal() - shifted_value) < eps ||
                fabs(inputs[regr]->get_signal() - shifted_norm) < eps) {
            cout << "[NOK] - Forgot to skip the regression column" << endl;
            return false;
        }

        if (fabs(inputs[regr]->get_signal() - norm) > eps) {
            cout << "[NOK] - Input data not initialised correctly" << endl;
            return false;
        }
    }

    return true;
}

bool OLP_test::test_computeHiddenStep() {
    double ax[size];
    for (int i = 0; i < size; i++)
        ax[i] = hidden[i]->get_axon()->get_signal();
    compute_hidden_step();
    bool result = true;
    for (int i = 0; i < size; i++)
        result = result && hidden[i]->get_axon()->get_signal() != ax[i];
    return result;
}

bool OLP_test::test_computeOutputStep(Dataset *data, int row, int regr) {
    std::vector<double> *mins = data->get_mins();
    std::vector<double> *maxs = data->get_maxs();
    double out = output->get_axon()->get_signal();
    double back = output->get_axon()->get_back_value();
    double den[size];
    for (int i = 0; i < size; i++)
        den[i] = output->get_dendrite(i)->get_back_value();
    double output_sig = compute_output_step(data, row, regr);
    if (fabs(output_sig - denormalize(output->get_axon()->get_signal(), data, regr) > eps)) {
        cout << "[NOK] - Incorrect output signal (probably forgot to set 'ret')" << endl;
        return false;
    }
    if (fabs(output->get_axon()->get_signal() - out) < eps) {
        cout << "[NOK] - Output signal not updated" << endl;
        return false;
    }
    if (fabs(output->get_axon()->get_back_value() - back) < eps) {
        cout << "[NOK] - Back propagation not initialised in the output neuron" << endl;
        return false;
    }
    if (fabs(output->get_axon()->get_back_value() - output->get_axon()->get_signal()) < eps) {
        cout << "[NOK] - Incorrectly initialised back propagation in the output neuron ('ret' instead of 'error'?)" << endl;
        return false;
    }
    for (int i = 0; i < size; i++)
        if (hidden[i]->get_axon()->get_back_value() == den[i]) {
            cout << "[NOK] - Back propagation through the output neuron not done" << endl;
            return false;
        }

    return true;
}

bool OLP_test::test_propagateBackHidden() {
    double ax[dim];
    for (int i = 0; i < dim; i++)
        ax[i] = inputs[i]->get_back_value();
    for (int i = 0; i < size; i++)
        hidden[i]->set_back_value(100);
    propagate_back_hidden();
    bool result = true;
    for (int i = 0; i < dim; i++)
        if (fabs(inputs[i]->get_back_value() - ax[i]) < eps) {
            result = false;
            break;
        }
    return result;
}

const double eps = 0.001;
const std::string default_path = "./grading/tests/";

double rel_error(double a, double b) {
    return fabs(a - b) / fabs(a);
}

template <typename T, typename... Arguments>
bool test_rel_error(std::ostream &out,
                      const std::string &function_name,
                      T result,
                      T expected,
                      T delta,
                      const Arguments&... args)
{
    bool success = (rel_error(result, expected) <= delta);

    out << (success ? "[SUCCESS] " : "[FAILURE] ");

    print_tested_function(out, function_name, args...);

    out << ": got " << result
        << " expected " << expected << "  The relative error should be in [-" << delta << "," << delta << "]";
    out << std::endl;

    return success;
}

int test_neuron_step(std::ostream &out, const std::string test_name)
{
    std::string entity_name = "Test neuron";
    start_test_suite(out, test_name);
    std::vector<int> res;

    cout << "Testing Neuron::step()...\t";
    std::vector<Node *> inputs({new Node(), new Node(), new Node()});
    Neuron neuron(
        3, inputs,
        [](double x) -> double { return x; },
        [](double x) -> double { return 1; });
    neuron.set_weight(0, 1);
    neuron.set_weight(1, 10);
    neuron.set_weight(2, 100);
    neuron.set_bias(1);

    Neuron neuron2(
        3, inputs,
        [](double x) -> double { return x + 1; },
        [](double x) -> double { return 1; });
    neuron2.set_weight(0, 1);
    neuron2.set_weight(1, 10);
    neuron2.set_weight(2, 100);
    neuron2.set_bias(1);

    inputs[0]->set_signal(10);
    inputs[1]->set_signal(100);
    inputs[2]->set_signal(1000);

    // Uncomment the following lines to debug your implementation
    // cout << "At neuron initialisation" << endl;
    // cout << neuron << endl;
    // cout << neuron2 << endl;

    neuron.step();
    neuron2.step();

    // Uncomment the following lines to debug your implementation
    // cout << "After one step" << endl;
    // cout << neuron << endl;
    // cout << neuron2 << endl;
    double n1s1 = neuron.get_axon()->get_signal();
    double n2s1 = neuron2.get_axon()->get_signal();
    double ci1s1 = neuron.get_collected_input();
    double ci2s1 = neuron2.get_collected_input();

    neuron.step();
    neuron2.step();

    // Uncomment the following lines to debug your implementation
    // cout << "After two steps" << endl;
    // cout << neuron << endl;
    // cout << neuron2 << endl;
    double n1s2 = neuron.get_axon()->get_signal();
    double n2s2 = neuron2.get_axon()->get_signal();
    double ci1s2 = neuron.get_collected_input();
    double ci2s2 = neuron2.get_collected_input();

    bool ex1ok = false;

    res.push_back(test_eq_approx(out, "Activation function used correctly?", n2s1, n1s1 + 1, eps));
    res.push_back(test_eq_approx(out, "Activation function used correctly?", n2s2, n1s2 + 1, eps));
    res.push_back(test_eq_approx(out, "Collected input initialised properly?", n1s1, n1s2, eps));
    res.push_back(test_eq_approx(out, "Collected input initialised properly?", n2s1, n2s2, eps));
    res.push_back(test_eq_approx(out, "Collected input computed properly (did you use a local variable instead of Neuron::collected_input?)?", ci1s1, n1s1, eps));
    res.push_back(test_eq_approx(out, "Collected input computed properly (did you use a local variable instead of Neuron::collected_input?)?", ci1s1, n1s1, eps));
    res.push_back(test_eq_approx(out, "Collected input computed properly (did you apply the activation function to the stored value?)?", n1s1, 101009.0, eps));

    switch (int(n1s1))
    {
    case 101009:
        ex1ok = true;
        break;

    case 101010:
        cout << "[NOK] - Bias not used properly" << endl;
        break;

    case 1109:
        cout << "[NOK] - Weights not used properly" << endl;
        break;

    case 1110:
        cout << "[NOK] - Bias and weights are not used properly" << endl;
        break;

    case 1009:
        cout << "[NOK] - Last dendrite not used properly" << endl;
        break;

    case 112:
        cout << "[NOK] - Signals not used properly" << endl;
        break;

    case 111:
        cout << "[NOK] - Bias and signals not used properly" << endl;
        break;

    default:
        cout << "[NOK] - Probably more than one error" << endl;
        break;
    }

        delete inputs[0];
        delete inputs[1];
        delete inputs[2];
        return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}

int test_neuron_step_back(std::ostream &out, const std::string test_name)
{
    std::string entity_name = "Test neuron step back";
    start_test_suite(out, test_name);
    std::vector<int> res;

    std::vector<Node *> inputs({new Node(), new Node(), new Node()});
    Neuron neuron(
        3, inputs,
        [](double x) -> double { return x; },
        [](double x) -> double { return 1; });
    neuron.set_weight(0, 1);
    neuron.set_weight(1, 10);
    neuron.set_weight(2, 100);
    neuron.set_bias(1);

    inputs[0]->set_signal(10);
    inputs[1]->set_signal(100);
    inputs[2]->set_signal(1000);
    neuron.step();
    neuron.step();

    neuron.set_weight(0, 1);
    neuron.set_weight(1, 10);
    neuron.set_weight(2, 100);
    neuron.set_bias(1);

    std::vector<Node *> inputs3({new Node(), new Node(), new Node()});
    inputs3[0]->set_signal(10);
    inputs3[1]->set_signal(100);
    inputs3[2]->set_signal(1000);

    Neuron neuron3(
        3, inputs3,
        [](double x) -> double { return x*x; },
        [](double x) -> double { return 2*x; });
    neuron3.set_weight(0, 1);
    neuron3.set_weight(1, 10);
    neuron3.set_weight(2, 100);
    neuron3.set_bias(1);

    neuron.set_collected_input(5);
    neuron3.set_collected_input(6);
    neuron.set_learning_rate(3);
    neuron3.set_learning_rate(3);

    neuron.set_back_value(2);
    neuron3.set_back_value(2);

    // Uncomment the following lines to debug your implementation
    // cout << "At neuron initialisation" << endl;
    // cout << neuron << endl;
    // cout << neuron3 << endl << endl;

    inputs[2]->set_back_value(-2);
    neuron.step_back();
    neuron3.step_back();

    // Uncomment the following lines to debug your implementation
    // cout << "After one step backwards" << endl;
    // cout << neuron << endl;
    // cout << neuron3 << endl << endl;

    double n1w1[4];
    double n1back1[4];
    double n3w1[4];
    double n3back1[4];
    for (int i = 0; i < 3; i++) {
        n1w1[i+1] = neuron.get_weight(i);
        n3w1[i+1] = neuron3.get_weight(i);
        n1back1[i+1] = neuron.get_dendrite(i)->get_back_value();
        n3back1[i+1] = neuron3.get_dendrite(i)->get_back_value();
    }
    n1w1[0] = neuron.get_bias();
    n3w1[0] = neuron3.get_bias();
    n1back1[0] = neuron.get_bias_dendrite()->get_back_value();
    n3back1[0] = neuron3.get_bias_dendrite()->get_back_value();

    res.push_back(test_neq_approx(out, "Last dendrite updated properly?", n1w1[3], 100.0, eps));
    res.push_back(test_neq_approx(out, "Bias updated properly?", n1w1[0], 0.0, eps));
    res.push_back(test_neq_approx(out, "Error propagated to next level?", n1back1[1], 0.0, eps));
    res.push_back(test_neq_approx(out, "Forgot the weights during error propagation?", n1back1[2], 2.0, eps));
    res.push_back(test_neq_approx(out, "Forgot the weights during error propagation?", n3back1[2], 24.0, eps));
    res.push_back(test_neq_approx(out, "Forgot the axons back value during error propagation?", n1back1[2], 10.0, eps));
    res.push_back(test_neq_approx(out, "Forgot the axons back value during error propagation?", n1back1[2], 100.0, eps));
    res.push_back(test_neq_approx(out, "Do you add propagated error to the existing one?", n1back1[3], 198.0, eps));
    res.push_back(test_neq_approx(out, "Do you subtract propagated error from the existing one?", n1back1[3], -202.0, eps));

    res.push_back(test_neq_approx(out, "'=' instead of '-=' in weight update for n1w1[1]?", n1w1[1], 60.0, eps));
    res.push_back(test_neq_approx(out, "'=' instead of '-=' in weight update for n1w1[2]?", n1w1[2], 600.0, eps));
    res.push_back(test_neq_approx(out, "'=' instead of '-=' in weight update for n1w1[3]?", n1w1[3], 6000.0, eps));

    res.push_back(test_neq_approx(out, "'= -' (assignment) instead of '-=' (cumulative updtate) in weight update for n1w1[1]?", n1w1[1], -60.0, eps));
    res.push_back(test_neq_approx(out, "'= -' (assignment) instead of '-=' (cumulative updtate) in weight update for n1w1[2]?", n1w1[2], -600.0, eps));
    res.push_back(test_neq_approx(out, "'= -' (assignment) instead of '-=' (cumulative updtate) in weight update for n1w1[3]?", n1w1[3], -6000.0, eps));


    if (n1w1[1] == -19 && n1w1[2] == -190 && n1w1[3] == -1900)
        cout << "[NOK] - Forgot the learning rate in weight update?" << endl;
    else if (n1w1[1] == -29 && n1w1[2] == -290 && n1w1[3] == -2900)
        cout << "[NOK] - Forgot the axon back value in weight update?" << endl;
    else if (n1w1[1] == -5 && n1w1[2] == 4 && n1w1[3] == 94)
        cout << "[NOK] - Forgot the dendrite signal in weight update?" << endl;
    else if (n3w1[1] == -59 && n3w1[2] == -590 && n3w1[3] == -5900)
        cout << "[NOK] - Forgot the collected input in weight update?" << endl;
    else if (n3w1[1] == -359 && n3w1[2] == -3590 && n3w1[3] == -35900)
        cout << "[NOK] - Forgot the activation function derivative in weight update?" << endl;
    else if (n3w1[1] == -2159 && n3w1[2] == -21590 && n3w1[3] == -215900)
        cout << "[NOK] - Used the activation function instead of its derivative in weight update?" << endl;


    res.push_back(test_eq_approx(out, "Check values of n1w1[0]", n1w1[0], 7.0, eps));
    res.push_back(test_eq_approx(out, "Check values of n1w1[1]", n1w1[1], -59.0, eps));
    res.push_back(test_eq_approx(out, "Check values of n1w1[2]", n1w1[2], -590.0, eps));
    res.push_back(test_eq_approx(out, "Check values of n1w1[3]", n1w1[3], -5900.0, eps));

    res.push_back(test_eq_approx(out, "Check values of n3w1[0]", n3w1[0], 73.0, eps));
    res.push_back(test_eq_approx(out, "Check values of n3w1[1]", n3w1[1], -719.0, eps));
    res.push_back(test_eq_approx(out, "Check values of n3w1[2]", n3w1[2], -7190.0, eps));
    res.push_back(test_eq_approx(out, "Check values of n3w1[3]", n3w1[3], -71900.0, eps));

    res.push_back(test_eq_approx(out, "Check values of n1back1[0]", n1back1[0], 2.0, 2.0));
    res.push_back(test_eq_approx(out, "Check values of n1back1[1]", n1back1[1], 2.0, eps));
    res.push_back(test_eq_approx(out, "Check values of n1back1[2]", n1back1[2], 20.0, eps));
    res.push_back(test_eq_approx(out, "Check values of n1back1[3]", n1back1[3], 200.0, eps));

    res.push_back(test_eq_approx(out, "Check values of n3back1[0]", n3back1[0], 24.0, 24.0));
    res.push_back(test_eq_approx(out, "Check values of n3back1[1]", n3back1[1], 24.0, eps));
    res.push_back(test_eq_approx(out, "Check values of n3back1[2]", n3back1[2], 240.0, eps));
    res.push_back(test_eq_approx(out, "Check values of n3back1[3]", n3back1[3], 2400.0, eps));

    delete inputs[0];
    delete inputs[1];
    delete inputs[2];
    delete inputs3[0];
    delete inputs3[1];
    delete inputs3[2];

    return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}

int test_perceptron_construction(std::ostream &out, const std::string test_name)
{
    std::string entity_name = "Test perceptron construction / destruction";
    start_test_suite(out, test_name);
    std::vector<int> res;

    const int dim = 3;
    const int size = 5;
    const double rate = 1;
    const double decay = 0;

    cout << "Testing OneLayerPerceptron constructor...\n";
    OLP_test *perceptron = new OLP_test(dim, size, rate, decay, [](double x) -> double {return x*x;}, [](double x) -> double {return 2*x;});
    if (!perceptron->test_inputs())
    {
        cout << "[FAILURE] - Input nodes are not initialised correctly" << endl;
    } else {
        cout << "[SUCCESS] - Input nodes are initialised correctly" << endl;
    }
    res.push_back(perceptron->test_inputs());
    if (!perceptron->test_inputs())
    {
        cout << "[FAILURE] - Hidden layer not initialised correctly" << endl;
    } else {
        cout << "[SUCCESS] - Hidden layer correctly initialised" << endl;
    }
    res.push_back(OLP_test::interpret_neuron_test(perceptron->test_hidden(), "A hidden"));
    if (!perceptron->test_inputs())
    {
        cout << "[FAILURE] - Output layer not initialised correctly" << endl;
    } else {
        cout << "[SUCCESS] - Output layer correctly initialised" << endl;
    }
    res.push_back(OLP_test::interpret_neuron_test(perceptron->test_output(), "The output"));
    if (!perceptron->test_inputs())
    {
        cout << "[FAILURE] - Activation not initialised correctly" << endl;
    } else {
        cout << "[SUCCESS] - Activation correctly initialised" << endl;
    }
    res.push_back(perceptron->test_act([](double x) -> double {return x*x;}, [](double x) -> double {return 2*x;}));

    cout << "\nTesting OneLayerPerceptron destructor...\n";

    delete perceptron;

    switch (Neuron::get_count())
    {
    case 1:
        cout << "[FAILURE] - Output neuron was not deleted" << endl;

    case 0:
        break;

    default:
        cout << "[FAILURE] - Hidden neurons were not deleted" << endl;
    }

    switch (Node::get_count())
    {
    case dim:
        cout << "[FAILURE] - Inputs were not deleted" << endl;

    case 0:
        cout << "[SUCCESS]" << endl;
        res.push_back(true);
        break;

    default:
        cout << "[FAILURE] - Some nodes were not deleted" << endl;
    }

    return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}

int test_perceptron_run(std::ostream &out, const std::string test_name)
{
    std::string entity_name = "Test perceptron prepare_inputs, compute_hidden_step, compute_output_step, propagate_back_hidden";
    start_test_suite(out, test_name);
    std::vector<int> res;

    cout << "Preparing to test `OneLayerPerceptron::run`..." << endl;
    std::ifstream ftest("csv/test.csv");
    if (ftest.fail())
    {
        std::cout << "\tCould not read from the file `" << "csv/test.csv" << "` - make sure its available"
                  << std::endl;
    }

    Dataset test(ftest);
    const int dim = test.get_dim() - 1;
    const int count = test.get_nb_samples();
    const int row = 1;
    cout << "\tRead test data from " << "csv/test.csv" << endl;
    cout << "\t" << count << " rows of dimension " << test.get_dim()
         << endl;

    const int regr = 2;
    const int size = default_nb_neurons;
    const double rate = 1;
    const double decay = 0;

    cout << "\tSetting up a perceptron for testing...\t";
    OLP_test perceptron(dim, size, rate, decay);
    cout << "done." << endl;

    cout << "...preparation done." << endl;

    cout << "Testing OneLayerPerceptron::prepare_inputs()...\t\t";
    if (perceptron.test_prepareInputs(&test, row, regr)) {
        cout << "[SUCCESS]" << endl;
        res.push_back(true);
    } else {
        cout << "[FAILURE]" << endl;
        res.push_back(false);
    }
    cout << "Testing OneLayerPerceptron::compute_hidden_step()...\t";
    if (!perceptron.test_computeHiddenStep()) {
        cout << "[FAILURE]" << endl;
        res.push_back(false);
    }
    else {
        cout << "[SUCCESS]" << endl;
        res.push_back(true);
    }
    cout << "Testing OneLayerPerceptron::compute_output_step()...\t";
    if (perceptron.test_computeOutputStep(&test, row, regr)) {
        cout << "[SUCCESS]" << endl;
        res.push_back(true);
    }
    else {
        cout << "[FAILURE]" << endl;
        res.push_back(false);
    }
    cout << "Testing OneLayerPerceptron::propagate_back_hidden()...\t";
    if (!perceptron.test_propagateBackHidden()) {
        cout << "[FAILURE]" << endl;
        res.push_back(false);
    }
    else {
        cout << "[SUCCESS]" << endl;
        res.push_back(true);
    }
    return end_test_suite(out, test_name, accumulate(res.begin(), res.end(), 0), res.size());
}

int grading(std::ostream &out, const int test_case_number)
{
/**

Annotations used for the autograder.

[START-AUTOGRADER-ANNOTATION]
{
  "total" : 4,
  "names" : [
        "neuron.cpp::test_neuron_step",
        "neuron.cpp::test_neuron_step_back",
        "perceptron.cpp::test_perceptron_construction",
        "perceptron.cpp::test_perceptron_run"
        ],
  "points" : [10, 10, 10, 10]
}
[END-AUTOGRADER-ANNOTATION]
*/

    int const total_test_cases = 4;
    std::string const test_names[total_test_cases] = {"test_neuron_step", "test_neuron_step_back", "test_perceptron_construction", "test_perceptron_run"};
    int const points[total_test_cases] = {10, 10, 10, 10};
    int (*test_functions[total_test_cases]) (std::ostream &, const std::string) = {
      test_neuron_step, test_neuron_step_back, test_perceptron_construction, test_perceptron_run
    };

    return run_grading(out, test_case_number, total_test_cases,
                       test_names, points,
                       test_functions);
}

} // End of namepsace tdgrading
