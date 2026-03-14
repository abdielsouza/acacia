#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

// Include CV headers
#include "acacia/cv/kernel.hpp"
#include "acacia/cv/filters.hpp"

void test_kernel_creation()
{
    std::cout << "Testing Kernel Creation..." << std::endl;

    // Test Gaussian kernel
    auto gaussian_kernel = acacia::cv::Kernel<double>::gaussian(3, 1.0);
    assert(gaussian_kernel.rows() == 3);
    assert(gaussian_kernel.cols() == 3);

    // Test Sobel kernels
    auto sobel_x = acacia::cv::Kernel<double>::sobel_x();
    auto sobel_y = acacia::cv::Kernel<double>::sobel_y();
    assert(sobel_x.rows() == 3);
    assert(sobel_y.rows() == 3);

    std::cout << "Kernel Creation test passed!" << std::endl;
}

void test_convolution()
{
    std::cout << "Testing Convolution..." << std::endl;

    // Simple 3x3 image
    std::vector<std::vector<double>> image = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    // Identity kernel
    std::vector<std::vector<double>> identity_data = {
        {0, 0, 0},
        {0, 1, 0},
        {0, 0, 0}
    };
    acacia::cv::Kernel<double> identity_kernel(identity_data);

    auto result = acacia::cv::convolve(image, identity_kernel);

    // Should be the same as input
    for (size_t i = 0; i < image.size(); ++i) {
        for (size_t j = 0; j < image[i].size(); ++j) {
            assert(std::abs(result[i][j] - image[i][j]) < 1e-6);
        }
    }

    std::cout << "Convolution test passed!" << std::endl;
}

void test_gaussian_blur()
{
    std::cout << "Testing Gaussian Blur..." << std::endl;

    // Simple image
    std::vector<std::vector<double>> image = {
        {0, 0, 0, 0, 0},
        {0, 0, 1, 0, 0},
        {0, 1, 1, 1, 0},
        {0, 0, 1, 0, 0},
        {0, 0, 0, 0, 0}
    };

    auto blurred = acacia::cv::gaussian_blur(image, 3, 1.0);

    // Check dimensions
    assert(blurred.size() == image.size());
    assert(blurred[0].size() == image[0].size());

    // Center should be blurred (less sharp)
    assert(blurred[2][2] < image[2][2]);

    std::cout << "Gaussian Blur test passed!" << std::endl;
}

void test_sobel_edge_detection()
{
    std::cout << "Testing Sobel Edge Detection..." << std::endl;

    // Simple edge image
    std::vector<std::vector<double>> image = {
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 1, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0}
    };

    auto edges = acacia::cv::sobel_edge_detection(image);

    // Check dimensions
    assert(edges.size() == image.size());
    assert(edges[0].size() == image[0].size());

    // Should detect edges
    bool has_edges = false;
    for (const auto& row : edges) {
        for (double val : row) {
            if (val > 0) has_edges = true;
        }
    }
    assert(has_edges);

    std::cout << "Sobel Edge Detection test passed!" << std::endl;
}

void test_median_filter()
{
    std::cout << "Testing Median Filter..." << std::endl;

    // Image with salt and pepper noise
    std::vector<std::vector<double>> noisy_image = {
        {0, 0, 0},
        {0, 1, 0},
        {0, 0, 0}
    };

    // Add noise
    noisy_image[0][0] = 1; // salt
    noisy_image[2][2] = 1; // pepper

    auto filtered = acacia::cv::median_filter(noisy_image, 3);

    // Check dimensions
    assert(filtered.size() == noisy_image.size());
    assert(filtered[0].size() == noisy_image[0].size());

    std::cout << "Median Filter test passed!" << std::endl;
}

void run_cv_tests()
{
    test_kernel_creation();
    test_convolution();
    test_gaussian_blur();
    test_sobel_edge_detection();
    test_median_filter();
}