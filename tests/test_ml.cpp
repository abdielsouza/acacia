#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

// Include ML headers
#include "acacia/ml/regression.hpp"
#include "acacia/ml/loss.hpp"
#include "acacia/ml/metrics.hpp"
#include "acacia/ml/optimization.hpp"

void test_linear_regression()
{
    std::cout << "Testing Linear Regression..." << std::endl;

    // Simple dataset: y = 2*x + 1
    std::vector<std::vector<double>> X = {{1}, {2}, {3}, {4}, {5}};
    std::vector<double> y = {3, 5, 7, 9, 11};

    acacia::ml::LinearRegression<double> model;
    model.fit(X, y);

    auto predictions = model.predict(X);

    // Check predictions are close to actual values
    for (size_t i = 0; i < predictions.size(); ++i) {
        assert(std::abs(predictions[i] - y[i]) < 0.1);
    }

    std::cout << "Linear Regression test passed!" << std::endl;
}

void test_mse_loss()
{
    std::cout << "Testing MSE Loss..." << std::endl;

    std::vector<double> predictions = {1.0, 2.0, 3.0};
    std::vector<double> targets = {1.1, 2.1, 2.9};

    double loss = acacia::ml::MSELoss::loss(predictions, targets);
    assert(loss > 0);

    auto gradients = acacia::ml::MSELoss::gradient(predictions, targets);
    assert(gradients.size() == 3);

    std::cout << "MSE Loss test passed!" << std::endl;
}

void test_mse_metric()
{
    std::cout << "Testing MSE Metric..." << std::endl;

    std::vector<double> predictions = {1.0, 2.0, 3.0};
    std::vector<double> targets = {1.1, 2.1, 2.9};

    double mse = acacia::ml::MSE::evaluate(predictions, targets);
    assert(mse > 0);

    std::cout << "MSE Metric test passed!" << std::endl;
}

void test_sgd_optimizer()
{
    std::cout << "Testing SGD Optimizer..." << std::endl;

    std::vector<double> params = {1.0, 2.0};
    std::vector<double> gradients = {0.1, 0.2};

    acacia::ml::SGD<double> optimizer(0.1);
    optimizer.step(params, gradients);

    assert(std::abs(params[0] - 0.99) < 1e-6);
    assert(std::abs(params[1] - 1.98) < 1e-6);

    std::cout << "SGD Optimizer test passed!" << std::endl;
}

void run_ml_tests()
{
    test_linear_regression();
    test_mse_loss();
    test_mse_metric();
    test_sgd_optimizer();
}