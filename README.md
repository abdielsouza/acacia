# Acacia - C++ Scientific Computing Library

Acacia is a modern C++23 library for scientific computing, providing modules for machine learning, computer vision, statistics, data handling, and mathematical utilities.

## Features

### Machine Learning (ML)
- **Regression**: Linear Regression, Polynomial Regression, Ridge Regression
- **Loss Functions**: MSE, MAE, Binary Cross-Entropy, Categorical Cross-Entropy
- **Metrics**: MSE, RMSE, MAE, R², Accuracy, Precision, Recall, F1-Score
- **Optimization**: SGD, SGD with Momentum, Adam, RMSProp

### Computer Vision (CV)
- **Kernels**: Gaussian, Sobel, Laplacian, Sharpen, Box blur
- **Filters**: Convolution, Gaussian blur, Edge detection (Sobel, Laplacian), Median filter, Bilateral filter

### Statistics
- **Distributions**: Normal, Uniform, Bernoulli, Categorical, Exponential
- Sampling and PDF/CDF computation

### Data Handling
- **Dataset Readers**: CSV, Excel (placeholder), Parquet (placeholder), Database (placeholder)
- **Dataset Writers**: CSV, Excel (placeholder), Parquet (placeholder), Database (placeholder)

### Utilities
- **Linear Algebra**: Vectors, Matrices with basic operations
- **Calculus**: Numerical differentiation, integration, root finding

## Building

This project uses [xmake](https://xmake.io) as the build system.

```bash
# Build the library
xmake

# Run tests
xmake run acacia-tests

# Install
xmake install
```

## Usage

### Machine Learning Example

```cpp
#include "acacia/ml/regression.hpp"
#include "acacia/ml/loss.hpp"
#include "acacia/ml/metrics.hpp"

int main() {
    // Training data
    std::vector<std::vector<double>> X = {{1}, {2}, {3}, {4}, {5}};
    std::vector<double> y = {2, 4, 6, 8, 10};

    // Create and train model
    acacia::ml::LinearRegression<double> model;
    model.fit(X, y);

    // Make predictions
    auto predictions = model.predict(X);

    // Evaluate
    double mse = acacia::ml::MSE::evaluate(predictions, y);
    return 0;
}
```

### Computer Vision Example

```cpp
#include "acacia/cv/filters.hpp"
#include "acacia/cv/kernel.hpp"

int main() {
    // Create a simple image (5x5)
    std::vector<std::vector<double>> image = {
        {0, 0, 0, 0, 0},
        {0, 0, 1, 0, 0},
        {0, 1, 1, 1, 0},
        {0, 0, 1, 0, 0},
        {0, 0, 0, 0, 0}
    };

    // Apply Gaussian blur
    auto blurred = acacia::cv::gaussian_blur(image, 3, 1.0);

    // Detect edges
    auto edges = acacia::cv::sobel_edge_detection(image);

    return 0;
}
```

### Statistics Example

```cpp
#include "acacia/stats/distribution.hpp"

int main() {
    // Create a normal distribution
    acacia::stats::Normal dist(0.0, 1.0);

    // Sample values
    double sample = dist.sample();
    auto samples = dist.sample(100);

    // Compute PDF/CDF
    double pdf_val = dist.pdf(0.0);
    double cdf_val = dist.cdf(0.0);

    return 0;
}
```

## Requirements

- C++23 compatible compiler (GCC 11+, Clang 14+, MSVC 2022+)
- [xmake](https://xmake.io) build system

## Architecture

The library is organized into the following namespaces:

- `acacia::ml` - Machine Learning algorithms
- `acacia::cv` - Computer Vision operations
- `acacia::stats` - Statistical distributions and functions
- `acacia::dataset` - Data loading and saving utilities
- `acacia::utils` - Mathematical utilities and linear algebra

## Contributing

Contributions are welcome! Please ensure code follows C++23 standards and includes appropriate tests.

## License

This project is open source. Please check the license file for details.