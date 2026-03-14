#ifndef ACACIA_ML_OPTIMIZATION_HPP_
#define ACACIA_ML_OPTIMIZATION_HPP_

#pragma once

#include <vector>
#include <functional>
#include <cmath>
#include <algorithm>
#include "acacia/ml/concepts.hpp"

namespace acacia::ml
{
    /**
     * @brief Abstract base class for optimization algorithms.
     */
    template<Numeric T>
    class Optimizer
    {
    public:
        virtual ~Optimizer() = default;

        /**
         * @brief Updates the parameters using the gradients.
         * @param params The current parameters to update.
         * @param gradients The gradients corresponding to each parameter.
         */
        virtual void step(std::vector<T>& params, const std::vector<T>& gradients) = 0;

        /**
         * @brief Resets the optimizer state (useful for momentum-based optimizers).
         */
        virtual void reset() {}
    };

    /**
     * @brief Stochastic Gradient Descent (SGD) optimizer.
     *
     * Basic gradient descent algorithm that updates parameters by moving in the
     * direction opposite to the gradient.
     */
    template<Numeric T>
    class SGD : public Optimizer<T>
    {
    private:
        T learning_rate_;

    public:
        /**
         * @brief Constructs an SGD optimizer.
         * @param learning_rate The step size for parameter updates. Default is 0.01.
         */
        explicit SGD(T learning_rate = static_cast<T>(0.01)) : learning_rate_(learning_rate) {}

        void step(std::vector<T>& params, const std::vector<T>& gradients) override
        {
            if (params.size() != gradients.size()) {
                throw std::invalid_argument("Parameters and gradients must have the same size");
            }

            for (size_t i = 0; i < params.size(); ++i) {
                params[i] -= learning_rate_ * gradients[i];
            }
        }
    };

    /**
     * @brief SGD with Momentum optimizer.
     *
     * Adds momentum to SGD to help accelerate convergence and reduce oscillations.
     */
    template<Numeric T>
    class SGDMomentum : public Optimizer<T>
    {
    private:
        T learning_rate_;
        T momentum_;
        std::vector<T> velocity_;

    public:
        /**
         * @brief Constructs an SGD with Momentum optimizer.
         * @param learning_rate The step size for parameter updates. Default is 0.01.
         * @param momentum The momentum factor. Default is 0.9.
         */
        SGDMomentum(T learning_rate = static_cast<T>(0.01), T momentum = static_cast<T>(0.9))
            : learning_rate_(learning_rate), momentum_(momentum) {}

        void step(std::vector<T>& params, const std::vector<T>& gradients) override
        {
            if (params.size() != gradients.size()) {
                throw std::invalid_argument("Parameters and gradients must have the same size");
            }

            if (velocity_.size() != params.size()) {
                velocity_.resize(params.size(), static_cast<T>(0));
            }

            for (size_t i = 0; i < params.size(); ++i) {
                velocity_[i] = momentum_ * velocity_[i] + learning_rate_ * gradients[i];
                params[i] -= velocity_[i];
            }
        }

        void reset() override
        {
            std::fill(velocity_.begin(), velocity_.end(), static_cast<T>(0));
        }
    };

    /**
     * @brief Adam (Adaptive Moment Estimation) optimizer.
     *
     * Combines the advantages of AdaGrad and RMSProp. Adapts learning rates for each
     * parameter and maintains moving averages of both the gradients and their squares.
     */
    template<Numeric T>
    class Adam : public Optimizer<T>
    {
    private:
        T learning_rate_;
        T beta1_;
        T beta2_;
        T epsilon_;
        std::vector<T> m_; // First moment estimate
        std::vector<T> v_; // Second moment estimate
        size_t t_; // Time step

    public:
        /**
         * @brief Constructs an Adam optimizer.
         * @param learning_rate The step size for parameter updates. Default is 0.001.
         * @param beta1 Exponential decay rate for the first moment estimates. Default is 0.9.
         * @param beta2 Exponential decay rate for the second moment estimates. Default is 0.999.
         * @param epsilon Small constant for numerical stability. Default is 1e-8.
         */
        Adam(T learning_rate = static_cast<T>(0.001),
             T beta1 = static_cast<T>(0.9),
             T beta2 = static_cast<T>(0.999),
             T epsilon = static_cast<T>(1e-8))
            : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), t_(0) {}

        void step(std::vector<T>& params, const std::vector<T>& gradients) override
        {
            if (params.size() != gradients.size()) {
                throw std::invalid_argument("Parameters and gradients must have the same size");
            }

            ++t_;

            if (m_.size() != params.size()) {
                m_.resize(params.size(), static_cast<T>(0));
                v_.resize(params.size(), static_cast<T>(0));
            }

            for (size_t i = 0; i < params.size(); ++i) {
                // Update biased first moment estimate
                m_[i] = beta1_ * m_[i] + (static_cast<T>(1) - beta1_) * gradients[i];

                // Update biased second moment estimate
                v_[i] = beta2_ * v_[i] + (static_cast<T>(1) - beta2_) * gradients[i] * gradients[i];

                // Compute bias-corrected first moment estimate
                T m_hat = m_[i] / (static_cast<T>(1) - std::pow(beta1_, t_));

                // Compute bias-corrected second moment estimate
                T v_hat = v_[i] / (static_cast<T>(1) - std::pow(beta2_, t_));

                // Update parameters
                params[i] -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
            }
        }

        void reset() override
        {
            std::fill(m_.begin(), m_.end(), static_cast<T>(0));
            std::fill(v_.begin(), v_.end(), static_cast<T>(0));
            t_ = 0;
        }
    };

    /**
     * @brief RMSProp optimizer.
     *
     * Adapts the learning rate for each parameter by dividing by a running average
     * of the magnitudes of recent gradients.
     */
    template<Numeric T>
    class RMSProp : public Optimizer<T>
    {
    private:
        T learning_rate_;
        T rho_;
        T epsilon_;
        std::vector<T> v_; // Running average of squared gradients

    public:
        /**
         * @brief Constructs an RMSProp optimizer.
         * @param learning_rate The step size for parameter updates. Default is 0.001.
         * @param rho Decay rate for the moving average. Default is 0.9.
         * @param epsilon Small constant for numerical stability. Default is 1e-8.
         */
        RMSProp(T learning_rate = static_cast<T>(0.001),
                T rho = static_cast<T>(0.9),
                T epsilon = static_cast<T>(1e-8))
            : learning_rate_(learning_rate), rho_(rho), epsilon_(epsilon) {}

        void step(std::vector<T>& params, const std::vector<T>& gradients) override
        {
            if (params.size() != gradients.size()) {
                throw std::invalid_argument("Parameters and gradients must have the same size");
            }

            if (v_.size() != params.size()) {
                v_.resize(params.size(), static_cast<T>(0));
            }

            for (size_t i = 0; i < params.size(); ++i) {
                // Update running average of squared gradients
                v_[i] = rho_ * v_[i] + (static_cast<T>(1) - rho_) * gradients[i] * gradients[i];

                // Update parameters
                params[i] -= learning_rate_ * gradients[i] / (std::sqrt(v_[i]) + epsilon_);
            }
        }

        void reset() override
        {
            std::fill(v_.begin(), v_.end(), static_cast<T>(0));
        }
    };
}

#endif