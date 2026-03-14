#ifndef ACACIA_UTILS_CALCULUS_HPP_
#define ACACIA_UTILS_CALCULUS_HPP_

#pragma once

#include <functional>
#include <cmath>
#include <vector>
#include <concepts>

namespace acacia::utils
{
    /// Concept for a callable that takes a value of type T and returns a value of type T
    template<typename Func, typename T>
    concept UnaryFunction = std::invocable<Func, T> && 
                            std::convertible_to<std::invoke_result_t<Func, T>, T>;

    /// Numerical derivative using central difference approximation
    template<typename Func, typename T>
    requires UnaryFunction<Func, T>
    T derivative(Func f, T x, T h = static_cast<T>(1e-5)) {
        return (f(x + h) - f(x - h)) / (static_cast<T>(2) * h);
    }

    /// Numerical integral using trapezoidal rule
    template<typename Func, typename T>
    requires UnaryFunction<Func, T>
    T integral(Func f, T a, T b, size_t n = 1000) {
        T h = (b - a) / static_cast<T>(n);
        T sum = (f(a) + f(b)) / static_cast<T>(2);
        for(size_t i = 1; i < n; ++i) {
            sum += f(a + static_cast<T>(i) * h);
        }
        return sum * h;
    }

    /// Simple root finder using bisection method
    template<typename Func, typename T>
    requires UnaryFunction<Func, T>
    T find_root(Func f, T a, T b, T tol = static_cast<T>(1e-6)) {
        T fa = f(a), fb = f(b);
        if (fa * fb > static_cast<T>(0)) return static_cast<T>(0); // no root in interval
        while (std::abs(b - a) > tol) {
            T c = (a + b) / static_cast<T>(2);
            T fc = f(c);
            if (fc == static_cast<T>(0)) return c;
            if (fa * fc < static_cast<T>(0)) b = c;
            else a = c;
        }
        return (a + b) / static_cast<T>(2);
    }
}

#endif