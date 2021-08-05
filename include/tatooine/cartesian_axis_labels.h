#ifndef TATOOINE_CARTESIAN_AXIS_LABELS_H
#define TATOOINE_CARTESIAN_AXIS_LABELS_H
//==============================================================================
#include <string>
//==============================================================================
namespace tatooine{
//==============================================================================
template <size_t I>
struct cartesian_axis_label_impl;
template <size_t I>
static auto constexpr cartesian_axis_label =
    cartesian_axis_label_impl<I>::value;
template <>
struct cartesian_axis_label_impl<0> {
  static constexpr auto value = std::string_view{"x"};
};
template <>
struct cartesian_axis_label_impl<1> {
  static constexpr auto value = std::string_view{"y"};
};
template <>
struct cartesian_axis_label_impl<2> {
  static constexpr auto value = std::string_view{"z"};
};

//==============================================================================
}
//==============================================================================
#endif
