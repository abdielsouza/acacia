#ifndef ACACIA_CV_KERNEL_HPP_
#define ACACIA_CV_KERNEL_HPP_

#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <concepts>

namespace acacia::cv
{
    /// Concept for numeric types
    template<typename T>
    concept Numeric = std::is_arithmetic_v<T>;

    /**
     * @brief Represents a 2D convolution kernel.
     *
     * A kernel is a small matrix used for image processing operations like
     * blurring, sharpening, edge detection, etc.
     */
    template<Numeric T>
    class Kernel
    {
    private:
        std::vector<std::vector<T>> data_;
        size_t rows_;
        size_t cols_;

    public:
        /**
         * @brief Default constructor.
         */
        Kernel() : rows_(0), cols_(0) {}

        /**
         * @brief Constructs a kernel from a 2D vector.
         * @param kernel_data The kernel data as a 2D vector.
         */
        explicit Kernel(const std::vector<std::vector<T>>& kernel_data)
        {
            if (kernel_data.empty() || kernel_data[0].empty()) {
                throw std::invalid_argument("Kernel data cannot be empty");
            }

            size_t cols = kernel_data[0].size();
            for (const auto& row : kernel_data) {
                if (row.size() != cols) {
                    throw std::invalid_argument("All rows must have the same number of columns");
                }
            }

            data_ = kernel_data;
            rows_ = kernel_data.size();
            cols_ = cols;
        }

        /**
         * @brief Constructs a kernel with given dimensions and fills with zeros.
         * @param rows Number of rows.
         * @param cols Number of columns.
         */
        Kernel(size_t rows, size_t cols) : rows_(rows), cols_(cols), data_(rows, std::vector<T>(cols, 0)) {}

        /**
         * @brief Access operator.
         * @param row Row index.
         * @param col Column index.
         * @return Reference to the element.
         */
        T& operator()(size_t row, size_t col)
        {
            if (row >= rows_ || col >= cols_) {
                throw std::out_of_range("Index out of bounds");
            }
            return data_[row][col];
        }

        /**
         * @brief Const access operator.
         * @param row Row index.
         * @param col Column index.
         * @return Const reference to the element.
         */
        const T& operator()(size_t row, size_t col) const
        {
            if (row >= rows_ || col >= cols_) {
                throw std::out_of_range("Index out of bounds");
            }
            return data_[row][col];
        }

        /**
         * @brief Returns the number of rows.
         * @return Number of rows.
         */
        size_t rows() const { return rows_; }

        /**
         * @brief Returns the number of columns.
         * @return Number of columns.
         */
        size_t cols() const { return cols_; }

        /**
         * @brief Returns the kernel data.
         * @return Const reference to the 2D vector.
         */
        const std::vector<std::vector<T>>& data() const { return data_; }

        /**
         * @brief Normalizes the kernel so that all elements sum to 1.
         */
        void normalize()
        {
            T sum = 0;
            for (const auto& row : data_) {
                for (const auto& val : row) {
                    sum += val;
                }
            }

            if (sum != 0) {
                for (auto& row : data_) {
                    for (auto& val : row) {
                        val /= sum;
                    }
                }
            }
        }

        /**
         * @brief Creates a Gaussian kernel.
         * @param size Size of the kernel (must be odd).
         * @param sigma Standard deviation of the Gaussian.
         * @return Gaussian kernel.
         */
        static Kernel<T> gaussian(size_t size, T sigma = 1.0)
        {
            if (size % 2 == 0) {
                throw std::invalid_argument("Kernel size must be odd");
            }

            Kernel<T> kernel(size, size);
            int center = size / 2;
            T sum = 0;

            for (int i = 0; i < static_cast<int>(size); ++i) {
                for (int j = 0; j < static_cast<int>(size); ++j) {
                    int x = i - center;
                    int y = j - center;
                    T value = std::exp(-(x * x + y * y) / (2 * sigma * sigma));
                    kernel(i, j) = value;
                    sum += value;
                }
            }

            // Normalize
            for (size_t i = 0; i < size; ++i) {
                for (size_t j = 0; j < size; ++j) {
                    kernel(i, j) /= sum;
                }
            }

            return kernel;
        }

        /**
         * @brief Creates a Sobel kernel for edge detection in X direction.
         * @return Sobel X kernel.
         */
        static Kernel<T> sobel_x()
        {
            Kernel<T> kernel(3, 3);
            kernel(0, 0) = -1; kernel(0, 1) = 0; kernel(0, 2) = 1;
            kernel(1, 0) = -2; kernel(1, 1) = 0; kernel(1, 2) = 2;
            kernel(2, 0) = -1; kernel(2, 1) = 0; kernel(2, 2) = 1;
            return kernel;
        }

        /**
         * @brief Creates a Sobel kernel for edge detection in Y direction.
         * @return Sobel Y kernel.
         */
        static Kernel<T> sobel_y()
        {
            Kernel<T> kernel(3, 3);
            kernel(0, 0) = -1; kernel(0, 1) = -2; kernel(0, 2) = -1;
            kernel(1, 0) =  0; kernel(1, 1) =  0; kernel(1, 2) =  0;
            kernel(2, 0) =  1; kernel(2, 1) =  2; kernel(2, 2) =  1;
            return kernel;
        }

        /**
         * @brief Creates a Laplacian kernel for edge detection.
         * @return Laplacian kernel.
         */
        static Kernel<T> laplacian()
        {
            Kernel<T> kernel(3, 3);
            kernel(0, 0) = 0; kernel(0, 1) = 1; kernel(0, 2) = 0;
            kernel(1, 0) = 1; kernel(1, 1) =-4; kernel(1, 2) = 1;
            kernel(2, 0) = 0; kernel(2, 1) = 1; kernel(2, 2) = 0;
            return kernel;
        }

        /**
         * @brief Creates a box blur kernel.
         * @param size Size of the kernel (must be odd).
         * @return Box blur kernel.
         */
        static Kernel<T> box_blur(size_t size)
        {
            if (size % 2 == 0) {
                throw std::invalid_argument("Kernel size must be odd");
            }

            Kernel<T> kernel(size, size);
            T value = static_cast<T>(1) / (size * size);

            for (size_t i = 0; i < size; ++i) {
                for (size_t j = 0; j < size; ++j) {
                    kernel(i, j) = value;
                }
            }

            return kernel;
        }

        /**
         * @brief Creates a sharpening kernel.
         * @return Sharpening kernel.
         */
        static Kernel<T> sharpen()
        {
            Kernel<T> kernel(3, 3);
            kernel(0, 0) =  0; kernel(0, 1) = -1; kernel(0, 2) =  0;
            kernel(1, 0) = -1; kernel(1, 1) =  5; kernel(1, 2) = -1;
            kernel(2, 0) =  0; kernel(2, 1) = -1; kernel(2, 2) =  0;
            return kernel;
        }
    };
}

#endif