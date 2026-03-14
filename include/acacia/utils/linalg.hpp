#ifndef ACACIA_UTILS_LINALG_HPP_
#define ACACIA_UTILS_LINALG_HPP_

#pragma once

#include <type_traits>
#include <array>
#include <cstdint>
#include <stdfloat>

namespace acacia::utils
{
    // =============== SECTION 1 ===============
    // The objects defined below are strong
    // models for vectors and matrices with
    // a dynamic implementation that is
    // efficient in terms of memory.
    // =========================================

    template <typename T>
    concept Numeric = std::is_arithmetic_v<T>;

    /// General definition for vectors.
    template <Numeric T, std::size_t N>
    using Vector = std::array<T, N>;

    // For 2D vectors.

    using Vec2u8    = Vector<std::uint8_t, 2>;
    using Vec2u16   = Vector<std::uint16_t, 2>;
    using Vec2u32   = Vector<std::uint32_t, 2>;
    using Vec2u64   = Vector<std::uint64_t, 2>;

    using Vec2i8    = Vector<std::int8_t, 2>;
    using Vec2i16   = Vector<std::int16_t, 2>;
    using Vec2i32   = Vector<std::int32_t, 2>;
    using Vec2i64   = Vector<std::int64_t, 2>;

    using Vec2f16   = Vector<std::float16_t, 2>;
    using Vec2f32   = Vector<std::float32_t, 2>;
    using Vec2f64   = Vector<std::float64_t, 2>;
    using Vec2f128  = Vector<std::float128_t, 2>;

    using Vec2d     = Vector<double, 2>;

    // For 3D vectors.

    using Vec3u8    = Vector<std::uint8_t, 3>;
    using Vec3u16   = Vector<std::uint16_t, 3>;
    using Vec3u32   = Vector<std::uint32_t, 3>;
    using Vec3u64   = Vector<std::uint64_t, 3>;

    using Vec3i8    = Vector<std::int8_t, 3>;
    using Vec3i16   = Vector<std::int16_t, 3>;
    using Vec3i32   = Vector<std::int32_t, 3>;
    using Vec3i64   = Vector<std::int64_t, 3>;

    using Vec3f16   = Vector<std::float16_t, 3>;
    using Vec3f32   = Vector<std::float32_t, 3>;
    using Vec3f64   = Vector<std::float64_t, 3>;
    using Vec3f128  = Vector<std::float128_t, 3>;

    using Vec3d     = Vector<double, 3>;

    // For 4D vectors.

    using Vec4u8    = Vector<std::uint8_t, 4>;
    using Vec4u16   = Vector<std::uint16_t, 4>;
    using Vec4u32   = Vector<std::uint32_t, 4>;
    using Vec4u64   = Vector<std::uint64_t, 4>;

    using Vec4i8    = Vector<std::int8_t, 4>;
    using Vec4i16   = Vector<std::int16_t, 4>;
    using Vec4i32   = Vector<std::int32_t, 4>;
    using Vec4i64   = Vector<std::int64_t, 4>;

    using Vec4f16   = Vector<std::float16_t, 4>;
    using Vec4f32   = Vector<std::float32_t, 4>;
    using Vec4f64   = Vector<std::float64_t, 4>;
    using Vec4f128  = Vector<std::float128_t, 4>;

    using Vec4d     = Vector<double, 4>;

    // ---------------------------------------------
    // Now the definitions for matrices.
    // ---------------------------------------------

    /**
     * A matrix is like a multi-dimensional vector
     * with rows and columns. It can be very useful
     * when representing a lot of data in ML model
     * training and CV pipeline.
     */
    template <Numeric T, std::size_t NRows, std::size_t NCols>
    class Matrix
    {
    private:
        std::array<T, NRows * NCols> data_;

    public:
        // Default constructor
        constexpr Matrix() = default;

        // Constructor from initializer list (row-major order)
        constexpr Matrix(std::initializer_list<std::initializer_list<T>> list)
        {
            auto row_it = list.begin();
            for (std::size_t i = 0; i < NRows; ++i) {
                auto col_it = row_it->begin();
                for (std::size_t j = 0; j < NCols; ++j) {
                    data_[i * NCols + j] = *col_it++;
                }
                ++row_it;
            }
        }

        // Access operator (row, column)
        constexpr T& operator()(std::size_t row, std::size_t col)
        {
            return data_[row * NCols + col];
        }

        constexpr const T& operator()(std::size_t row, std::size_t col) const
        {
            return data_[row * NCols + col];
        }

        // Get dimensions
        static constexpr std::size_t rows() { return NRows; }
        static constexpr std::size_t cols() { return NCols; }

        // Raw data access
        constexpr T* data() { return data_.data(); }
        constexpr const T* data() const { return data_.data(); }
    };
}

#endif