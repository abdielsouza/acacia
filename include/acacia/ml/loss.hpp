#ifndef ACACIA_ML_LOSS_HPP_
#define ACACIA_ML_LOSS_HPP_

#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include "acacia/ml/concepts.hpp"

namespace acacia::ml
{
    /**
     * @brief Mean Squared Error (MSE) loss function.
     *
     * Commonly used for regression tasks. Measures the average squared difference
     * between predicted and actual values.
     */
    class MSELoss
    {
    public:
        template<Numeric T>
        static T loss(const std::vector<T>& predictions, const std::vector<T>& targets)
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

        template<Numeric T>
        static std::vector<T> gradient(const std::vector<T>& predictions, const std::vector<T>& targets)
        {
            if (predictions.size() != targets.size()) {
                throw std::invalid_argument("Predictions and targets must have the same size");
            }

            std::vector<T> grad(predictions.size());
            for (size_t i = 0; i < predictions.size(); ++i) {
                grad[i] = 2 * (predictions[i] - targets[i]) / static_cast<T>(predictions.size());
            }
            return grad;
        }
    };

    /**
     * @brief Mean Absolute Error (MAE) loss function.
     *
     * Also known as L1 loss. Less sensitive to outliers than MSE.
     */
    class MAELoss
    {
    public:
        template<Numeric T>
        static T loss(const std::vector<T>& predictions, const std::vector<T>& targets)
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

        template<Numeric T>
        static std::vector<T> gradient(const std::vector<T>& predictions, const std::vector<T>& targets)
        {
            if (predictions.size() != targets.size()) {
                throw std::invalid_argument("Predictions and targets must have the same size");
            }

            std::vector<T> grad(predictions.size());
            for (size_t i = 0; i < predictions.size(); ++i) {
                T diff = predictions[i] - targets[i];
                grad[i] = (diff > 0 ? 1 : (diff < 0 ? -1 : 0)) / static_cast<T>(predictions.size());
            }
            return grad;
        }
    };

    /**
     * @brief Binary Cross-Entropy loss function.
     *
     * Used for binary classification tasks. Measures the performance of a classification
     * model whose output is a probability value between 0 and 1.
     */
    class BinaryCrossEntropyLoss
    {
    public:
        template<Numeric T>
        static T loss(const std::vector<T>& predictions, const std::vector<T>& targets)
        {
            if (predictions.size() != targets.size()) {
                throw std::invalid_argument("Predictions and targets must have the same size");
            }

            T sum = 0;
            const T epsilon = static_cast<T>(1e-15); // To avoid log(0)

            for (size_t i = 0; i < predictions.size(); ++i) {
                T pred = std::clamp(predictions[i], epsilon, static_cast<T>(1) - epsilon);
                T target = targets[i];
                sum += -target * std::log(pred) - (static_cast<T>(1) - target) * std::log(static_cast<T>(1) - pred);
            }
            return sum / static_cast<T>(predictions.size());
        }

        template<Numeric T>
        static std::vector<T> gradient(const std::vector<T>& predictions, const std::vector<T>& targets)
        {
            if (predictions.size() != targets.size()) {
                throw std::invalid_argument("Predictions and targets must have the same size");
            }

            std::vector<T> grad(predictions.size());
            const T epsilon = static_cast<T>(1e-15);

            for (size_t i = 0; i < predictions.size(); ++i) {
                T pred = std::clamp(predictions[i], epsilon, static_cast<T>(1) - epsilon);
                T target = targets[i];
                grad[i] = (pred - target) / (pred * (static_cast<T>(1) - pred) * static_cast<T>(predictions.size()));
            }
            return grad;
        }
    };

    /**
     * @brief Categorical Cross-Entropy loss function.
     *
     * Used for multi-class classification tasks. Measures the performance of a classification
     * model whose output is a probability distribution over multiple classes.
     */
    class CategoricalCrossEntropyLoss
    {
    public:
        template<Numeric T>
        static T loss(const std::vector<std::vector<T>>& predictions, const std::vector<std::vector<T>>& targets)
        {
            if (predictions.size() != targets.size() || predictions.empty() || predictions[0].size() != targets[0].size()) {
                throw std::invalid_argument("Predictions and targets must have the same dimensions");
            }

            T sum = 0;
            const T epsilon = static_cast<T>(1e-15);

            for (size_t i = 0; i < predictions.size(); ++i) {
                for (size_t j = 0; j < predictions[i].size(); ++j) {
                    T pred = std::clamp(predictions[i][j], epsilon, static_cast<T>(1) - epsilon);
                    T target = targets[i][j];
                    sum += -target * std::log(pred);
                }
            }
            return sum / static_cast<T>(predictions.size());
        }

        template<Numeric T>
        static std::vector<std::vector<T>> gradient(const std::vector<std::vector<T>>& predictions, const std::vector<std::vector<T>>& targets)
        {
            if (predictions.size() != targets.size() || predictions.empty() || predictions[0].size() != targets[0].size()) {
                throw std::invalid_argument("Predictions and targets must have the same dimensions");
            }

            std::vector<std::vector<T>> grad(predictions.size(), std::vector<T>(predictions[0].size()));
            const T epsilon = static_cast<T>(1e-15);

            for (size_t i = 0; i < predictions.size(); ++i) {
                for (size_t j = 0; j < predictions[i].size(); ++j) {
                    T pred = std::clamp(predictions[i][j], epsilon, static_cast<T>(1) - epsilon);
                    grad[i][j] = (pred - targets[i][j]) / static_cast<T>(predictions.size());
                }
            }
            return grad;
        }
    };
}

#endif