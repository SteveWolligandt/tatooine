#ifndef TATOOINE_CONCPETS_H
#define TATOOINE_CONCPETS_H
//==============================================================================
#include <tatooine/invocable_with_n_types.h>
#include <tatooine/type_traits.h>

#include <concepts>
#include <ranges>
//==============================================================================
namespace tatooine {
//==============================================================================
// typedefs
//==============================================================================
template <typename T0, typename T1>
concept same_as = std::same_as<T0, T1>;
//------------------------------------------------------------------------------
template <typename T, typename... Ts>
concept either_of = (same_as<T, Ts> || ...);
//------------------------------------------------------------------------------
template <typename T>
concept integral = std::integral<T>;
//------------------------------------------------------------------------------
template <typename T>
concept signed_integral = std::signed_integral<T>;
//------------------------------------------------------------------------------
template <typename T>
concept unsigned_integral = std::unsigned_integral<T>;
//------------------------------------------------------------------------------
template <typename T>
concept floating_point = std::floating_point<T>;
//------------------------------------------------------------------------------
template <typename T>
concept arithmetic = integral<T> || floating_point<T>;
//------------------------------------------------------------------------------
template <typename T>
concept arithmetic_or_complex = arithmetic<T> || is_complex<T>;
//------------------------------------------------------------------------------
template <typename From, typename To>
concept convertible_to = std::convertible_to<From, To>;
//------------------------------------------------------------------------------
template <typename From>
concept convertible_to_floating_point =
  convertible_to<From, float> ||
  convertible_to<From, double> ||
  convertible_to<From, long double>;
//------------------------------------------------------------------------------
template <typename From>
concept convertible_to_integral =
  convertible_to<From, bool> ||
  convertible_to<From, char> ||
  convertible_to<From, unsigned char> ||
  convertible_to<From, char8_t> ||
  convertible_to<From, char16_t> ||
  convertible_to<From, char32_t> ||
  convertible_to<From, wchar_t> ||
  convertible_to<From, short> ||
  convertible_to<From, unsigned short> ||
  convertible_to<From, int> ||
  convertible_to<From, unsigned int> ||
  convertible_to<From, long> ||
  convertible_to<From, unsigned long> ||
  convertible_to<From, long long> ||
  convertible_to<From, unsigned long long>;
//------------------------------------------------------------------------------
template <typename T>
concept has_defined_real_type = requires {
  typename T::real_type;
};
//------------------------------------------------------------------------------
template <typename T>
concept has_defined_iterator = requires {
  typename T::iterator;
};
//------------------------------------------------------------------------------
template <typename T>
concept has_defined_this_type = requires {
  typename T::this_type;
};
//------------------------------------------------------------------------------
template <typename T>
concept has_defined_parent_type = requires {
  typename T::parent_type;
};
//------------------------------------------------------------------------------
template <typename T>
concept has_defined_tensor_type = requires {
  typename T::tensor_type;
};
//------------------------------------------------------------------------------
template <typename T>
concept has_defined_pos_type = requires {
  typename T::pos_type;
};
//==============================================================================
// ranges etc.
//==============================================================================
template <typename T>
concept forward_iterator = std::forward_iterator<T>;
//------------------------------------------------------------------------------
template <typename T>
concept bidirectional_iterator = std::bidirectional_iterator<T>;
//------------------------------------------------------------------------------
template <typename T>
concept range = std::ranges::range<T>;
//------------------------------------------------------------------------------
template <typename T>
concept arithmetic_range =
    range<T> && arithmetic<std::ranges::range_value_t<T>>;
//------------------------------------------------------------------------------
template <typename T>
concept integral_range =
    range<T> && integral<std::ranges::range_value_t<T>>;
//------------------------------------------------------------------------------
template <typename T>
concept floating_point_range =
    range<T> && floating_point<std::ranges::range_value_t<T>>;
//------------------------------------------------------------------------------
template <typename T, typename S>
concept range_of =
    range<T> && same_as<std::ranges::range_value_t<T>, S>;
//------------------------------------------------------------------------------
template <typename T, typename... Ss>
concept range_of_either =
    range<T> && either_of<std::ranges::range_value_t<T>, Ss...>;
//==============================================================================
// indexable
//==============================================================================
template <typename T>
concept indexable = requires(T const t, std::size_t i) {
  { t[i] };
  { t.at(i) };
};
//==============================================================================
// methods
//==============================================================================
template <typename F, typename... Args>
concept invocable = std::invocable<F, Args...>;
//-----------------------------------------------------------------------------
template <typename F, typename... Args>
concept regular_invocable = std::regular_invocable<F, Args...>;
//-----------------------------------------------------------------------------
template <typename T>
concept has_static_num_dimensions_method = requires {
  { T::num_dimensions() } -> std::convertible_to<std::size_t>;
};
//-----------------------------------------------------------------------------
template <typename T>
concept has_static_rank_method = requires {
  { T::rank() } -> std::convertible_to<std::size_t>;
};
//-----------------------------------------------------------------------------
template <typename F, typename... Is>
concept invocable_with_integrals = std::invocable<F, Is...> &&
                                   (integral<Is> && ...);
//==============================================================================
// misc
//==============================================================================
template <typename Reader, typename Readable>
concept can_read = requires(Reader reader, Readable readable) {
  reader.read(readable);
};
//------------------------------------------------------------------------------
template <typename T>
concept has_real_type = requires {typename T::real_type;};
//------------------------------------------------------------------------------
template <typename T>
concept has_pos_type = requires {typename T::pos_type;};
//------------------------------------------------------------------------------
template <typename T>
concept has_tensor_type = requires {typename T::tensor_type;};
//------------------------------------------------------------------------------
template <typename T>
concept has_num_dimensions = requires {
  { T::num_dimensions } -> integral;
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
