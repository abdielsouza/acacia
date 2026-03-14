#ifndef ACACIA_CV_FILTERS_HPP_
#define ACACIA_CV_FILTERS_HPP_

#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <concepts>
#include "acacia/cv/kernel.hpp"

namespace acacia::cv
{


    /**
     * @brief Applies a convolution filter to an image using a given kernel.
     *
     * This function performs 2D convolution on a grayscale image. The image is
     * represented as a 2D vector where each element is a pixel value.
     *
     * @param image Input image as a 2D vector (rows x cols).
     * @param kernel Convolution kernel.
     * @return Filtered image.
     */
    template<Numeric T>
    std::vector<std::vector<T>> convolve(const std::vector<std::vector<T>>& image, const Kernel<T>& kernel)
    {
        if (image.empty() || image[0].empty()) {
            throw std::invalid_argument("Image cannot be empty");
        }

        size_t image_rows = image.size();
        size_t image_cols = image[0].size();
        size_t kernel_rows = kernel.rows();
        size_t kernel_cols = kernel.cols();

        // Ensure kernel dimensions are odd
        if (kernel_rows % 2 == 0 || kernel_cols % 2 == 0) {
            throw std::invalid_argument("Kernel dimensions must be odd");
        }

        // Output image
        std::vector<std::vector<T>> output(image_rows, std::vector<T>(image_cols, 0));

        int kernel_center_row = kernel_rows / 2;
        int kernel_center_col = kernel_cols / 2;

        // Perform convolution
        for (size_t i = 0; i < image_rows; ++i) {
            for (size_t j = 0; j < image_cols; ++j) {
                T sum = 0;

                // Apply kernel
                for (size_t ki = 0; ki < kernel_rows; ++ki) {
                    for (size_t kj = 0; kj < kernel_cols; ++kj) {
                        int image_i = static_cast<int>(i) + (static_cast<int>(ki) - kernel_center_row);
                        int image_j = static_cast<int>(j) + (static_cast<int>(kj) - kernel_center_col);

                        // Check bounds
                        if (image_i >= 0 && image_i < static_cast<int>(image_rows) &&
                            image_j >= 0 && image_j < static_cast<int>(image_cols)) {
                            sum += image[image_i][image_j] * kernel(ki, kj);
                        }
                    }
                }

                output[i][j] = sum;
            }
        }

        return output;
    }

    /**
     * @brief Applies Gaussian blur to an image.
     *
     * @param image Input image.
     * @param kernel_size Size of the Gaussian kernel (must be odd).
     * @param sigma Standard deviation of the Gaussian.
     * @return Blurred image.
     */
    template<Numeric T>
    std::vector<std::vector<T>> gaussian_blur(const std::vector<std::vector<T>>& image,
                                              size_t kernel_size = 5,
                                              T sigma = 1.0)
    {
        auto kernel = Kernel<T>::gaussian(kernel_size, sigma);
        return convolve(image, kernel);
    }

    /**
     * @brief Applies box blur to an image.
     *
     * @param image Input image.
     * @param kernel_size Size of the box kernel (must be odd).
     * @return Blurred image.
     */
    template<Numeric T>
    std::vector<std::vector<T>> box_blur(const std::vector<std::vector<T>>& image, size_t kernel_size = 5)
    {
        auto kernel = Kernel<T>::box_blur(kernel_size);
        return convolve(image, kernel);
    }

    /**
     * @brief Detects edges in an image using the Sobel operator.
     *
     * @param image Input image.
     * @return Edge magnitude image.
     */
    template<Numeric T>
    std::vector<std::vector<T>> sobel_edge_detection(const std::vector<std::vector<T>>& image)
    {
        if (image.empty() || image[0].empty()) {
            throw std::invalid_argument("Image cannot be empty");
        }

        size_t rows = image.size();
        size_t cols = image[0].size();

        auto kernel_x = Kernel<T>::sobel_x();
        auto kernel_y = Kernel<T>::sobel_y();

        auto grad_x = convolve(image, kernel_x);
        auto grad_y = convolve(image, kernel_y);

        std::vector<std::vector<T>> edges(rows, std::vector<T>(cols, 0));

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                T gx = grad_x[i][j];
                T gy = grad_y[i][j];
                edges[i][j] = std::sqrt(gx * gx + gy * gy);
            }
        }

        return edges;
    }

    /**
     * @brief Applies Laplacian edge detection to an image.
     *
     * @param image Input image.
     * @return Edge image.
     */
    template<Numeric T>
    std::vector<std::vector<T>> laplacian_edge_detection(const std::vector<std::vector<T>>& image)
    {
        auto kernel = Kernel<T>::laplacian();
        return convolve(image, kernel);
    }

    /**
     * @brief Sharpens an image using a sharpening kernel.
     *
     * @param image Input image.
     * @return Sharpened image.
     */
    template<Numeric T>
    std::vector<std::vector<T>> sharpen(const std::vector<std::vector<T>>& image)
    {
        auto kernel = Kernel<T>::sharpen();
        return convolve(image, kernel);
    }

    /**
     * @brief Applies median filter to reduce noise.
     *
     * @param image Input image.
     * @param kernel_size Size of the median kernel (must be odd).
     * @return Filtered image.
     */
    template<Numeric T>
    std::vector<std::vector<T>> median_filter(const std::vector<std::vector<T>>& image, size_t kernel_size = 3)
    {
        if (image.empty() || image[0].empty()) {
            throw std::invalid_argument("Image cannot be empty");
        }

        if (kernel_size % 2 == 0) {
            throw std::invalid_argument("Kernel size must be odd");
        }

        size_t rows = image.size();
        size_t cols = image[0].size();
        size_t half_kernel = kernel_size / 2;

        std::vector<std::vector<T>> output(rows, std::vector<T>(cols, 0));

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                std::vector<T> neighborhood;

                // Collect neighborhood values
                for (size_t ki = 0; ki < kernel_size; ++ki) {
                    for (size_t kj = 0; kj < kernel_size; ++kj) {
                        int ni = static_cast<int>(i) + static_cast<int>(ki) - static_cast<int>(half_kernel);
                        int nj = static_cast<int>(j) + static_cast<int>(kj) - static_cast<int>(half_kernel);

                        if (ni >= 0 && ni < static_cast<int>(rows) &&
                            nj >= 0 && nj < static_cast<int>(cols)) {
                            neighborhood.push_back(image[ni][nj]);
                        }
                    }
                }

                // Sort and find median
                std::sort(neighborhood.begin(), neighborhood.end());
                size_t median_idx = neighborhood.size() / 2;
                output[i][j] = neighborhood[median_idx];
            }
        }

        return output;
    }

    /**
     * @brief Applies bilateral filter for edge-preserving smoothing.
     *
     * @param image Input image.
     * @param spatial_sigma Standard deviation for spatial Gaussian.
     * @param intensity_sigma Standard deviation for intensity Gaussian.
     * @param kernel_size Size of the kernel.
     * @return Filtered image.
     */
    template<Numeric T>
    std::vector<std::vector<T>> bilateral_filter(const std::vector<std::vector<T>>& image,
                                                 T spatial_sigma = 1.0,
                                                 T intensity_sigma = 1.0,
                                                 size_t kernel_size = 5)
    {
        if (image.empty() || image[0].empty()) {
            throw std::invalid_argument("Image cannot be empty");
        }

        if (kernel_size % 2 == 0) {
            throw std::invalid_argument("Kernel size must be odd");
        }

        size_t rows = image.size();
        size_t cols = image[0].size();
        size_t half_kernel = kernel_size / 2;

        std::vector<std::vector<T>> output(rows, std::vector<T>(cols, 0));

        // Pre-compute spatial Gaussian
        std::vector<std::vector<T>> spatial_weights(kernel_size, std::vector<T>(kernel_size));
        for (size_t i = 0; i < kernel_size; ++i) {
            for (size_t j = 0; j < kernel_size; ++j) {
                int x = static_cast<int>(i) - static_cast<int>(half_kernel);
                int y = static_cast<int>(j) - static_cast<int>(half_kernel);
                spatial_weights[i][j] = std::exp(-(x * x + y * y) / (2 * spatial_sigma * spatial_sigma));
            }
        }

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                T center_value = image[i][j];
                T weighted_sum = 0;
                T weight_sum = 0;

                for (size_t ki = 0; ki < kernel_size; ++ki) {
                    for (size_t kj = 0; kj < kernel_size; ++kj) {
                        int ni = static_cast<int>(i) + static_cast<int>(ki) - static_cast<int>(half_kernel);
                        int nj = static_cast<int>(j) + static_cast<int>(kj) - static_cast<int>(half_kernel);

                        if (ni >= 0 && ni < static_cast<int>(rows) &&
                            nj >= 0 && nj < static_cast<int>(cols)) {
                            T neighbor_value = image[ni][nj];
                            T intensity_diff = neighbor_value - center_value;
                            T intensity_weight = std::exp(-(intensity_diff * intensity_diff) /
                                                         (2 * intensity_sigma * intensity_sigma));

                            T total_weight = spatial_weights[ki][kj] * intensity_weight;
                            weighted_sum += neighbor_value * total_weight;
                            weight_sum += total_weight;
                        }
                    }
                }

                output[i][j] = weight_sum > 0 ? weighted_sum / weight_sum : center_value;
            }
        }

        return output;
    }
}

#endif