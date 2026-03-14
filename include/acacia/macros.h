#ifndef ACACIA_MACROS_HPP_
#define ACACIA_MACROS_HPP_

#pragma once

#if defined(_WIN32)
#define ACACIA_API __dllexport
#else
#define ACACIA_API __attribute__((visibility("default")))
#endif

#ifdef ACACIA_MACROS_UNLOCK_ALL

#endif

#endif