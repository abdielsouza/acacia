#include "acacia/stats/distribution.hpp"

// Normal distribution implementations
acacia::stats::Normal::Normal(double mean, double stddev, unsigned int seed)
    : mean_(mean), stddev_(stddev), dist_(mean, stddev), gen_(seed) {}

double acacia::stats::Normal::pdf(double x) const
{
    double diff = x - mean_;
    return (1.0 / (stddev_ * std::sqrt(2 * M_PI))) *
           std::exp(-0.5 * (diff * diff) / (stddev_ * stddev_));
}

double acacia::stats::Normal::cdf(double x) const
{
    double z = (x - mean_) / (stddev_ * std::sqrt(2.0));
    return 0.5 * (1.0 + std::erf(z));
}

double acacia::stats::Normal::sample()
{
    return dist_(gen_);
}

std::vector<double> acacia::stats::Normal::sample(size_t n)
{
    std::vector<double> samples(n);
    for (size_t i = 0; i < n; ++i) {
        samples[i] = dist_(gen_);
    }
    return samples;
}

// Uniform distribution implementations
acacia::stats::Uniform::Uniform(double a, double b, unsigned int seed)
    : a_(a), b_(b), dist_(a, b), gen_(seed)
{
    if (a >= b) {
        throw std::invalid_argument("Lower bound must be less than upper bound");
    }
}

double acacia::stats::Uniform::pdf(double x) const
{
    if (x >= a_ && x <= b_) {
        return 1.0 / (b_ - a_);
    }
    return 0.0;
}

double acacia::stats::Uniform::cdf(double x) const
{
    if (x < a_) return 0.0;
    if (x > b_) return 1.0;
    return (x - a_) / (b_ - a_);
}

double acacia::stats::Uniform::sample()
{
    return dist_(gen_);
}

std::vector<double> acacia::stats::Uniform::sample(size_t n)
{
    std::vector<double> samples(n);
    for (size_t i = 0; i < n; ++i) {
        samples[i] = dist_(gen_);
    }
    return samples;
}

// Bernoulli distribution implementations
acacia::stats::Bernoulli::Bernoulli(double p, unsigned int seed)
    : p_(p), dist_(p), gen_(seed)
{
    if (p < 0.0 || p > 1.0) {
        throw std::invalid_argument("Probability must be between 0 and 1");
    }
}

double acacia::stats::Bernoulli::pdf(double x) const
{
    if (x == 0.0) return 1.0 - p_;
    if (x == 1.0) return p_;
    return 0.0;
}

double acacia::stats::Bernoulli::cdf(double x) const
{
    if (x < 0.0) return 0.0;
    if (x < 1.0) return 1.0 - p_;
    return 1.0;
}

double acacia::stats::Bernoulli::sample()
{
    return dist_(gen_) ? 1.0 : 0.0;
}

std::vector<double> acacia::stats::Bernoulli::sample(size_t n)
{
    std::vector<double> samples(n);
    for (size_t i = 0; i < n; ++i) {
        samples[i] = dist_(gen_) ? 1.0 : 0.0;
    }
    return samples;
}

// Categorical distribution implementations
acacia::stats::Categorical::Categorical(const std::vector<double>& probabilities, unsigned int seed)
    : probabilities_(probabilities), dist_(probabilities.begin(), probabilities.end()), gen_(seed)
{
    double sum = 0.0;
    for (double p : probabilities) {
        if (p < 0.0) throw std::invalid_argument("Probabilities must be non-negative");
        sum += p;
    }
    if (sum == 0.0) throw std::invalid_argument("At least one probability must be positive");
    // Normalize
    for (double& p : probabilities_) {
        p /= sum;
    }
}

double acacia::stats::Categorical::pdf(double x) const
{
    int idx = static_cast<int>(x);
    if (idx >= 0 && idx < static_cast<int>(probabilities_.size())) {
        return probabilities_[idx];
    }
    return 0.0;
}

double acacia::stats::Categorical::cdf(double x) const
{
    int idx = static_cast<int>(x);
    double cum = 0.0;
    for (int i = 0; i <= idx && i < static_cast<int>(probabilities_.size()); ++i) {
        cum += probabilities_[i];
    }
    return cum;
}

double acacia::stats::Categorical::sample()
{
    return static_cast<double>(dist_(gen_));
}

std::vector<double> acacia::stats::Categorical::sample(size_t n)
{
    std::vector<double> samples(n);
    for (size_t i = 0; i < n; ++i) {
        samples[i] = static_cast<double>(dist_(gen_));
    }
    return samples;
}

// Exponential distribution implementations
acacia::stats::Exponential::Exponential(double lambda, unsigned int seed)
    : lambda_(lambda), dist_(lambda), gen_(seed)
{
    if (lambda <= 0.0) {
        throw std::invalid_argument("Rate parameter must be positive");
    }
}

double acacia::stats::Exponential::pdf(double x) const
{
    if (x >= 0.0) {
        return lambda_ * std::exp(-lambda_ * x);
    }
    return 0.0;
}

double acacia::stats::Exponential::cdf(double x) const
{
    if (x < 0.0) return 0.0;
    return 1.0 - std::exp(-lambda_ * x);
}

double acacia::stats::Exponential::sample()
{
    return dist_(gen_);
}

std::vector<double> acacia::stats::Exponential::sample(size_t n)
{
    std::vector<double> samples(n);
    for (size_t i = 0; i < n; ++i) {
        samples[i] = dist_(gen_);
    }
    return samples;
}
