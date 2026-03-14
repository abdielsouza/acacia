#ifndef ACACIA_ML_METRICS_HPP_
#define ACACIA_ML_METRICS_HPP_

#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include "acacia/ml/concepts.hpp"

namespace acacia::ml
{
    /**
     * @brief Mean Squared Error (MSE) metric.
     *
     * Measures the average squared difference between predicted and actual values.
     * Lower values indicate better performance.
     */
    class MSE
    {
    public:
        template<Numeric T>
        static T evaluate(const std::vector<T>& predictions, const std::vector<T>& targets)
        {
            if (predictions.size() != targets.size()) {
                throw std::invalid_argument("Predictions and targets must have the same size");
            }

            T sum = 0;
            for (size_t i = 0; i < predictions.size(); ++i) {
                T diff = predictions[i] - targets[i];
                sum += diff * diff;
            }
            return sum / static_cast<T>(predictions.size());
        }
    };

    /**
     * @brief Root Mean Squared Error (RMSE) metric.
     *
     * Square root of MSE. Provides error in the same units as the target variable.
     */
    class RMSE
    {
    public:
        template<Numeric T>
        static T evaluate(const std::vector<T>& predictions, const std::vector<T>& targets)
        {
            return std::sqrt(MSE::evaluate(predictions, targets));
        }
    };

    /**
     * @brief Mean Absolute Error (MAE) metric.
     *
     * Measures the average absolute difference between predicted and actual values.
     * Less sensitive to outliers than MSE.
     */
    class MAE
    {
    public:
        template<Numeric T>
        static T evaluate(const std::vector<T>& predictions, const std::vector<T>& targets)
        {
            if (predictions.size() != targets.size()) {
                throw std::invalid_argument("Predictions and targets must have the same size");
            }

            T sum = 0;
            for (size_t i = 0; i < predictions.size(); ++i) {
                sum += std::abs(predictions[i] - targets[i]);
            }
            return sum / static_cast<T>(predictions.size());
        }
    };

    /**
     * @brief R-squared (R²) metric.
     *
     * Measures the proportion of variance in the dependent variable that's predictable
     * from the independent variables. Values closer to 1 indicate better fit.
     */
    class RSquared
    {
    public:
        template<Numeric T>
        static T evaluate(const std::vector<T>& predictions, const std::vector<T>& targets)
        {
            if (predictions.size() != targets.size()) {
                throw std::invalid_argument("Predictions and targets must have the same size");
            }

            // Calculate mean of targets
            T mean_target = std::accumulate(targets.begin(), targets.end(), static_cast<T>(0)) / static_cast<T>(targets.size());

            // Calculate total sum of squares
            T ss_tot = 0;
            for (const auto& target : targets) {
                T diff = target - mean_target;
                ss_tot += diff * diff;
            }

            // Calculate residual sum of squares
            T ss_res = 0;
            for (size_t i = 0; i < predictions.size(); ++i) {
                T diff = targets[i] - predictions[i];
                ss_res += diff * diff;
            }

            // Avoid division by zero
            if (ss_tot == 0) {
                return static_cast<T>(1);
            }

            return static_cast<T>(1) - (ss_res / ss_tot);
        }
    };

    /**
     * @brief Accuracy metric for classification.
     *
     * Measures the proportion of correct predictions.
     */
    class Accuracy
    {
    public:
        template<Numeric T>
        static T evaluate(const std::vector<T>& predictions, const std::vector<T>& targets)
        {
            if (predictions.size() != targets.size()) {
                throw std::invalid_argument("Predictions and targets must have the same size");
            }

            size_t correct = 0;
            for (size_t i = 0; i < predictions.size(); ++i) {
                if (predictions[i] == targets[i]) {
                    ++correct;
                }
            }
            return static_cast<T>(correct) / static_cast<T>(predictions.size());
        }
    };

    /**
     * @brief Precision metric for binary classification.
     *
     * Measures the proportion of true positive predictions among all positive predictions.
     */
    class Precision
    {
    public:
        static double evaluate(const std::vector<int>& predictions, const std::vector<int>& targets)
        {
            if (predictions.size() != targets.size()) {
                throw std::invalid_argument("Predictions and targets must have the same size");
            }

            size_t true_positives = 0;
            size_t false_positives = 0;

            for (size_t i = 0; i < predictions.size(); ++i) {
                if (predictions[i] == 1) {
                    if (targets[i] == 1) {
                        ++true_positives;
                    } else {
                        ++false_positives;
                    }
                }
            }

            if (true_positives + false_positives == 0) {
                return 0.0; // Avoid division by zero
            }

            return static_cast<double>(true_positives) / static_cast<double>(true_positives + false_positives);
        }
    };

    /**
     * @brief Recall metric for binary classification.
     *
     * Measures the proportion of true positive predictions among all actual positives.
     */
    class Recall
    {
    public:
        static double evaluate(const std::vector<int>& predictions, const std::vector<int>& targets)
        {
            if (predictions.size() != targets.size()) {
                throw std::invalid_argument("Predictions and targets must have the same size");
            }

            size_t true_positives = 0;
            size_t false_negatives = 0;

            for (size_t i = 0; i < predictions.size(); ++i) {
                if (targets[i] == 1) {
                    if (predictions[i] == 1) {
                        ++true_positives;
                    } else {
                        ++false_negatives;
                    }
                }
            }

            if (true_positives + false_negatives == 0) {
                return 0.0; // Avoid division by zero
            }

            return static_cast<double>(true_positives) / static_cast<double>(true_positives + false_negatives);
        }
    };

    /**
     * @brief F1 Score for binary classification.
     *
     * Harmonic mean of precision and recall. Provides a balance between the two metrics.
     */
    class F1Score
    {
    public:
        static double evaluate(const std::vector<int>& predictions, const std::vector<int>& targets)
        {
            double precision = Precision::evaluate(predictions, targets);
            double recall = Recall::evaluate(predictions, targets);

            if (precision + recall == 0.0) {
                return 0.0; // Avoid division by zero
            }

            return 2.0 * (precision * recall) / (precision + recall);
        }
    };
}

#endif