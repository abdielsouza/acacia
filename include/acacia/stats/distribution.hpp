#ifndef ACACIA_STATS_DISTRIBUTION_HPP_
#define ACACIA_STATS_DISTRIBUTION_HPP_

#pragma once

#include <random>
#include <cmath>
#include <vector>
#include <stdexcept>

namespace acacia::stats
{
    /**
     * @brief Abstract base class for statistical distributions.
     *
     * This class provides a common interface for various probability distributions
     * commonly used in machine learning and deep learning applications. It defines
     * methods for computing probability density functions (PDF), cumulative distribution
     * functions (CDF), and generating random samples.
     */
    class Distribution
    {
    public:
        virtual ~Distribution() = default;

        /**
         * @brief Computes the probability density function (PDF) at a given point.
         * @param x The point at which to evaluate the PDF.
         * @return The probability density at x.
         */
        virtual double pdf(double x) const = 0;

        /**
         * @brief Computes the cumulative distribution function (CDF) at a given point.
         * @param x The point at which to evaluate the CDF.
         * @return The cumulative probability up to x.
         */
        virtual double cdf(double x) const = 0;

        /**
         * @brief Generates a single random sample from the distribution.
         * @return A random sample from the distribution.
         */
        virtual double sample() = 0;

        /**
         * @brief Generates multiple random samples from the distribution.
         * @param n The number of samples to generate.
         * @return A vector containing n random samples.
         */
        virtual std::vector<double> sample(size_t n) = 0;
    };

    /**
     * @brief Normal (Gaussian) distribution.
     *
     * The normal distribution is fundamental in statistics and machine learning,
     * particularly for modeling continuous random variables with symmetric bell-shaped
     * curves. It's widely used in variational autoencoders, Bayesian neural networks,
     * and as a prior distribution in probabilistic models.
     */
    class Normal : public Distribution
    {
    private:
        double mean_;
        double stddev_;
        mutable std::normal_distribution<double> dist_;
        mutable std::mt19937 gen_;

    public:
        /**
         * @brief Constructs a Normal distribution.
         * @param mean The mean (μ) of the distribution. Default is 0.0.
         * @param stddev The standard deviation (σ) of the distribution. Must be positive. Default is 1.0.
         * @param seed The seed for the random number generator. Default uses std::random_device.
         */
        Normal(double mean = 0.0, double stddev = 1.0, unsigned int seed = std::random_device{}());

        double pdf(double x) const override;
        double cdf(double x) const override;
        double sample() override;
        std::vector<double> sample(size_t n) override;
    };

    /**
     * @brief Uniform distribution (continuous).
     *
     * The uniform distribution assigns equal probability to all values within a specified
     * range. It's commonly used for random initialization of parameters, generating random
     * numbers in a fixed range, and as a non-informative prior in Bayesian analysis.
     */
    class Uniform : public Distribution
    {
    private:
        double a_; // lower bound
        double b_; // upper bound
        mutable std::uniform_real_distribution<double> dist_;
        mutable std::mt19937 gen_;

    public:
        /**
         * @brief Constructs a Uniform distribution.
         * @param a The lower bound of the distribution. Default is 0.0.
         * @param b The upper bound of the distribution. Must be greater than a. Default is 1.0.
         * @param seed The seed for the random number generator. Default uses std::random_device.
         * @throws std::invalid_argument if a >= b.
         */
        Uniform(double a = 0.0, double b = 1.0, unsigned int seed = std::random_device{}());

        double pdf(double x) const override;
        double cdf(double x) const override;
        double sample() override;
        std::vector<double> sample(size_t n) override;
    };

    /**
     * @brief Bernoulli distribution (for binary outcomes).
     *
     * The Bernoulli distribution models a single binary trial with two possible outcomes:
     * success (1) with probability p, or failure (0) with probability 1-p. It's essential
     * for binary classification, modeling coin flips, and implementing dropout in neural networks.
     */
    class Bernoulli : public Distribution
    {
    private:
        double p_; // probability of success
        mutable std::bernoulli_distribution dist_;
        mutable std::mt19937 gen_;

    public:
        /**
         * @brief Constructs a Bernoulli distribution.
         * @param p The probability of success (1). Must be between 0 and 1. Default is 0.5.
         * @param seed The seed for the random number generator. Default uses std::random_device.
         * @throws std::invalid_argument if p is not in [0, 1].
         */
        Bernoulli(double p = 0.5, unsigned int seed = std::random_device{}());

        double pdf(double x) const override;
        double cdf(double x) const override;
        double sample() override;
        std::vector<double> sample(size_t n) override;
    };

    /**
     * @brief Categorical distribution (multinomial with one trial).
     *
     * The categorical distribution is a generalization of the Bernoulli distribution to
     * multiple categories. It's used for multi-class classification, modeling discrete
     * choices, and as the output distribution in classification neural networks.
     */
    class Categorical : public Distribution
    {
    private:
        std::vector<double> probabilities_;
        mutable std::discrete_distribution<int> dist_;
        mutable std::mt19937 gen_;

    public:
        /**
         * @brief Constructs a Categorical distribution.
         * @param probabilities A vector of probabilities for each category. Will be normalized.
         * @param seed The seed for the random number generator. Default uses std::random_device.
         * @throws std::invalid_argument if probabilities contain negative values or all are zero.
         */
        Categorical(const std::vector<double>& probabilities, unsigned int seed = std::random_device{}());

        double pdf(double x) const override;
        double cdf(double x) const override;
        double sample() override;
        std::vector<double> sample(size_t n) override;

        /**
         * @brief Returns the number of categories in the distribution.
         * @return The number of categories.
         */
        size_t num_categories() const { return probabilities_.size(); }
    };

    /**
     * @brief Exponential distribution.
     *
     * The exponential distribution models the time between events in a Poisson process.
     * It's used in survival analysis, queuing theory, and as a prior for positive parameters
     * in Bayesian inference. The distribution is memoryless and has a decreasing hazard rate.
     */
    class Exponential : public Distribution
    {
    private:
        double lambda_; // rate parameter
        mutable std::exponential_distribution<double> dist_;
        mutable std::mt19937 gen_;

    public:
        /**
         * @brief Constructs an Exponential distribution.
         * @param lambda The rate parameter (λ). Must be positive. Default is 1.0.
         * @param seed The seed for the random number generator. Default uses std::random_device.
         * @throws std::invalid_argument if lambda <= 0.
         */
        Exponential(double lambda = 1.0, unsigned int seed = std::random_device{}());

        double pdf(double x) const override;
        double cdf(double x) const override;
        double sample() override;
        std::vector<double> sample(size_t n) override;
    };
}

#endif