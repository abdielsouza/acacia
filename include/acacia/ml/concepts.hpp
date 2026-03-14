#ifndef ACACIA_ML_CONCEPTS_HPP_
#define ACACIA_ML_CONCEPTS_HPP_

#pragma once

#include <concepts>

namespace acacia::ml
{
    /// Concept for numeric types.
    template<typename T>
    concept Numeric = std::is_arithmetic_v<T>;
}

#endif
