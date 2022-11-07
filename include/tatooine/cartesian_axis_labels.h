#ifndef TATOOINE_CARTESIAN_AXIS_LABELS_H
#define TATOOINE_CARTESIAN_AXIS_LABELS_H
//==============================================================================
#include <string>
//==============================================================================
namespace tatooine {
//==============================================================================
template <std::size_t I>
struct cartesian_axis_label_impl;
//==============================================================================
template <>
struct cartesian_axis_label_impl<0> {
  static constexpr auto value = std::string_view{"x"};
};
//==============================================================================
template <>
struct cartesian_axis_label_impl<1> {
  static constexpr auto value = std::string_view{"y"};
};
//==============================================================================
template <>
struct cartesian_axis_label_impl<2> {
  static constexpr auto value = std::string_view{"z"};
};
//==============================================================================
template <std::size_t I>
static auto constexpr cartesian_axis_label() {
  return cartesian_axis_label_impl<I>::value;
}
//==============================================================================
static auto constexpr cartesian_axis_label(std::size_t const i)
    -> std::string_view {
  switch (i) {
    case 0:
      return cartesian_axis_label<0>();
    case 1:
      return cartesian_axis_label<1>();
    case 2:
      return cartesian_axis_label<2>();
    default:
      return "";
  }
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
