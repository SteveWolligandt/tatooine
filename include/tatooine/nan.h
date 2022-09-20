#ifndef TATOOINE_NAN_H
#define TATOOINE_NAN_H
//==============================================================================
#include <cmath>
#include <tatooine/concepts.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <floating_point Float>
struct nan_impl;
template <>
struct nan_impl<float> {
  static auto value(const char* arg) { return std::nanf(arg); }
};
template <>
struct nan_impl<double> {
  static auto value(const char* arg) { return std::nan(arg); }
};
template <>
struct nan_impl<long double> {
  static auto value(const char* arg) { return std::nanl(arg); }
};

template <floating_point Float>
auto nan(const char* arg = "") {
  return nan_impl<Float>::value(arg);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
