#ifndef ACACIA_ML_REGRESSION_HPP_
#define ACACIA_ML_REGRESSION_HPP_

#pragma once

#include <vector>
#include <functional>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include "acacia/ml/concepts.hpp"
#include <stdexcept>

#include "acacia/ml/loss.hpp"
#include "acacia/ml/optimization.hpp"
#include "acacia/utils/linalg.hpp"

namespace acacia::ml
{
    /**
     * @brief Linear Regression model.
     *
     * Fits a linear relationship between input features and target values.
     * Uses ordinary least squares for training.
     */
    template<Numeric T>
    class LinearRegression
    {
    private:
        std::vector<T> weights_; // Including bias as the last element
        bool trained_ = false;

    public:
        /**
         * @brief Default constructor.
         */
        LinearRegression() = default;

        /**
         * @brief Fits the linear regression model to the training data.
         * @param X Matrix of input features (n_samples x n_features).
         * @param y Vector of target values (n_samples).
         */
        void fit(const std::vector<std::vector<T>>& X, const std::vector<T>& y)
        {
            if (X.empty() || y.empty() || X.size() != y.size()) {
                throw std::invalid_argument("Invalid input dimensions");
            }

            size_t n_samples = X.size();
            size_t n_features = X[0].size();

            // Add bias term (column of ones)
            std::vector<std::vector<T>> X_augmented(n_samples, std::vector<T>(n_features + 1));
            for (size_t i = 0; i < n_samples; ++i) {
                for (size_t j = 0; j < n_features; ++j) {
                    X_augmented[i][j] = X[i][j];
                }
                X_augmented[i][n_features] = static_cast<T>(1); // bias
            }

            // Normal equation: w = (X^T * X)^(-1) * X^T * y
            // For simplicity, we'll use a basic implementation
            // In practice, you might want to use more robust methods like SVD

            // Compute X^T * X
            std::vector<std::vector<T>> XtX(n_features + 1, std::vector<T>(n_features + 1, 0));
            for (size_t i = 0; i < n_features + 1; ++i) {
                for (size_t j = 0; j < n_features + 1; ++j) {
                    for (size_t k = 0; k < n_samples; ++k) {
                        XtX[i][j] += X_augmented[k][i] * X_augmented[k][j];
                    }
                }
            }

            // Compute X^T * y
            std::vector<T> Xty(n_features + 1, 0);
            for (size_t i = 0; i < n_features + 1; ++i) {
                for (size_t k = 0; k < n_samples; ++k) {
                    Xty[i] += X_augmented[k][i] * y[k];
                }
            }

            // Solve the system (simplified for small matrices)
            // For now, we'll use Gaussian elimination
            weights_ = solve_linear_system(XtX, Xty);
            trained_ = true;
        }

        /**
         * @brief Predicts target values for new input data.
         * @param X Matrix of input features.
         * @return Vector of predicted values.
         */
        std::vector<T> predict(const std::vector<std::vector<T>>& X) const
        {
            if (!trained_) {
                throw std::runtime_error("Model must be trained before prediction");
            }

            if (X.empty() || X[0].size() + 1 != weights_.size()) {
                throw std::invalid_argument("Invalid input dimensions");
            }

            std::vector<T> predictions(X.size());
            for (size_t i = 0; i < X.size(); ++i) {
                T pred = weights_.back(); // bias
                for (size_t j = 0; j < X[i].size(); ++j) {
                    pred += weights_[j] * X[i][j];
                }
                predictions[i] = pred;
            }
            return predictions;
        }

        /**
         * @brief Returns the learned weights (including bias).
         * @return Vector of weights.
         */
        const std::vector<T>& get_weights() const
        {
            if (!trained_) {
                throw std::runtime_error("Model must be trained before accessing weights");
            }
            return weights_;
        }

    private:
        // Simple Gaussian elimination for solving linear systems
        std::vector<T> solve_linear_system(std::vector<std::vector<T>> A, std::vector<T> b)
        {
            size_t n = A.size();

            // Forward elimination
            for (size_t i = 0; i < n; ++i) {
                // Find pivot
                size_t max_row = i;
                for (size_t k = i + 1; k < n; ++k) {
                    if (std::abs(A[k][i]) > std::abs(A[max_row][i])) {
                        max_row = k;
                    }
                }

                // Swap rows
                std::swap(A[i], A[max_row]);
                std::swap(b[i], b[max_row]);

                // Eliminate
                for (size_t k = i + 1; k < n; ++k) {
                    T factor = A[k][i] / A[i][i];
                    for (size_t j = i; j < n; ++j) {
                        A[k][j] -= factor * A[i][j];
                    }
                    b[k] -= factor * b[i];
                }
            }

            // Back substitution
            std::vector<T> x(n);
            for (int i = n - 1; i >= 0; --i) {
                x[i] = b[i];
                for (size_t j = i + 1; j < n; ++j) {
                    x[i] -= A[i][j] * x[j];
                }
                x[i] /= A[i][i];
            }

            return x;
        }
    };

    /**
     * @brief Polynomial Regression model.
     *
     * Fits a polynomial relationship between input features and target values.
     * Extends LinearRegression by transforming features to polynomial features.
     */
    template<Numeric T>
    class PolynomialRegression
    {
    private:
        LinearRegression<T> linear_model_;
        size_t degree_;
        bool trained_ = false;

    public:
        /**
         * @brief Constructs a polynomial regression model.
         * @param degree The degree of the polynomial. Default is 2.
         */
        explicit PolynomialRegression(size_t degree = 2) : degree_(degree) {}

        /**
         * @brief Fits the polynomial regression model to the training data.
         * @param X Matrix of input features (n_samples x n_features).
         * @param y Vector of target values (n_samples).
         */
        void fit(const std::vector<std::vector<T>>& X, const std::vector<T>& y)
        {
            if (X.empty() || y.empty() || X.size() != y.size()) {
                throw std::invalid_argument("Invalid input dimensions");
            }

            // Transform features to polynomial features
            std::vector<std::vector<T>> X_poly = generate_polynomial_features(X, degree_);
            linear_model_.fit(X_poly, y);
            trained_ = true;
        }

        /**
         * @brief Predicts target values for new input data.
         * @param X Matrix of input features.
         * @return Vector of predicted values.
         */
        std::vector<T> predict(const std::vector<std::vector<T>>& X) const
        {
            if (!trained_) {
                throw std::runtime_error("Model must be trained before prediction");
            }

            std::vector<std::vector<T>> X_poly = generate_polynomial_features(X, degree_);
            return linear_model_.predict(X_poly);
        }

        /**
         * @brief Returns the degree of the polynomial.
         * @return The polynomial degree.
         */
        size_t get_degree() const { return degree_; }

    private:
        std::vector<std::vector<T>> generate_polynomial_features(const std::vector<std::vector<T>>& X, size_t degree)
        {
            if (X.empty()) return {};

            size_t n_samples = X.size();
            size_t n_features = X[0].size();

            // Calculate number of polynomial features
            size_t n_poly_features = 0;
            for (size_t d = 0; d <= degree; ++d) {
                n_poly_features += static_cast<size_t>(std::pow(static_cast<double>(n_features), static_cast<double>(d)));
            }

            std::vector<std::vector<T>> X_poly(n_samples, std::vector<T>(n_poly_features, 0));

            // Generate polynomial features
            size_t feature_idx = 0;

            // Degree 0: constant term
            for (size_t i = 0; i < n_samples; ++i) {
                X_poly[i][feature_idx] = 1;
            }
            ++feature_idx;

            // Higher degrees
            for (size_t d = 1; d <= degree; ++d) {
                // Generate all combinations of features for degree d
                std::vector<size_t> indices(d, 0);
                while (true) {
                    // Compute product
                    for (size_t i = 0; i < n_samples; ++i) {
                        T product = 1;
                        for (size_t j = 0; j < d; ++j) {
                            product *= X[i][indices[j]];
                        }
                        X_poly[i][feature_idx] = product;
                    }
                    ++feature_idx;

                    // Next combination
                    size_t pos = d - 1;
                    while (pos >= 0) {
                        if (indices[pos] < n_features - 1) {
                            ++indices[pos];
                            for (size_t j = pos + 1; j < d; ++j) {
                                indices[j] = indices[pos];
                            }
                            break;
                        }
                        --pos;
                    }
                    if (pos < 0) break;
                }
            }

            return X_poly;
        }
    };

    /**
     * @brief Ridge Regression model.
     *
     * Linear regression with L2 regularization to prevent overfitting.
     */
    template<Numeric T>
    class RidgeRegression
    {
    private:
        std::vector<T> weights_;
        T alpha_; // Regularization parameter
        bool trained_ = false;

    public:
        /**
         * @brief Constructs a Ridge regression model.
         * @param alpha Regularization parameter. Default is 1.0.
         */
        explicit RidgeRegression(T alpha = static_cast<T>(1.0)) : alpha_(alpha) {}

        /**
         * @brief Fits the Ridge regression model to the training data.
         * @param X Matrix of input features (n_samples x n_features).
         * @param y Vector of target values (n_samples).
         */
        void fit(const std::vector<std::vector<T>>& X, const std::vector<T>& y)
        {
            if (X.empty() || y.empty() || X.size() != y.size()) {
                throw std::invalid_argument("Invalid input dimensions");
            }

            size_t n_samples = X.size();
            size_t n_features = X[0].size();

            // Add bias term
            std::vector<std::vector<T>> X_augmented(n_samples, std::vector<T>(n_features + 1));
            for (size_t i = 0; i < n_samples; ++i) {
                for (size_t j = 0; j < n_features; ++j) {
                    X_augmented[i][j] = X[i][j];
                }
                X_augmented[i][n_features] = static_cast<T>(1);
            }

            // Ridge regression: w = (X^T * X + αI)^(-1) * X^T * y
            std::vector<std::vector<T>> XtX(n_features + 1, std::vector<T>(n_features + 1, 0));
            for (size_t i = 0; i < n_features + 1; ++i) {
                for (size_t j = 0; j < n_features + 1; ++j) {
                    for (size_t k = 0; k < n_samples; ++k) {
                        XtX[i][j] += X_augmented[k][i] * X_augmented[k][j];
                    }
                }
                // Add regularization (except for bias term)
                if (i < n_features) {
                    XtX[i][i] += alpha_;
                }
            }

            std::vector<T> Xty(n_features + 1, 0);
            for (size_t i = 0; i < n_features + 1; ++i) {
                for (size_t k = 0; k < n_samples; ++k) {
                    Xty[i] += X_augmented[k][i] * y[k];
                }
            }

            weights_ = solve_linear_system(XtX, Xty);
            trained_ = true;
        }

        /**
         * @brief Predicts target values for new input data.
         * @param X Matrix of input features.
         * @return Vector of predicted values.
         */
        std::vector<T> predict(const std::vector<std::vector<T>>& X) const
        {
            if (!trained_) {
                throw std::runtime_error("Model must be trained before prediction");
            }

            if (X.empty() || X[0].size() + 1 != weights_.size()) {
                throw std::invalid_argument("Invalid input dimensions");
            }

            std::vector<T> predictions(X.size());
            for (size_t i = 0; i < X.size(); ++i) {
                T pred = weights_.back();
                for (size_t j = 0; j < X[i].size(); ++j) {
                    pred += weights_[j] * X[i][j];
                }
                predictions[i] = pred;
            }
            return predictions;
        }

        /**
         * @brief Returns the learned weights.
         * @return Vector of weights.
         */
        const std::vector<T>& get_weights() const
        {
            if (!trained_) {
                throw std::runtime_error("Model must be trained before accessing weights");
            }
            return weights_;
        }

    private:
        // Same as LinearRegression
        std::vector<T> solve_linear_system(std::vector<std::vector<T>> A, std::vector<T> b)
        {
            size_t n = A.size();

            for (size_t i = 0; i < n; ++i) {
                size_t max_row = i;
                for (size_t k = i + 1; k < n; ++k) {
                    if (std::abs(A[k][i]) > std::abs(A[max_row][i])) {
                        max_row = k;
                    }
                }

                std::swap(A[i], A[max_row]);
                std::swap(b[i], b[max_row]);

                for (size_t k = i + 1; k < n; ++k) {
                    T factor = A[k][i] / A[i][i];
                    for (size_t j = i; j < n; ++j) {
                        A[k][j] -= factor * A[i][j];
                    }
                    b[k] -= factor * b[i];
                }
            }

            std::vector<T> x(n);
            for (int i = n - 1; i >= 0; --i) {
                x[i] = b[i];
                for (size_t j = i + 1; j < n; ++j) {
                    x[i] -= A[i][j] * x[j];
                }
                x[i] /= A[i][i];
            }

            return x;
        }
    };
}

#endif