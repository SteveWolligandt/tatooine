#ifndef TATOOINE_FIELD_CONCEPT_H
#define TATOOINE_FIELD_CONCEPT_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/vec.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename T, typename Real, std::size_t NumDimensions>
concept weak_field_concept = requires(T f, vec<Real, NumDimensions> x, Real t) {
  { f(x, t) } ;
};
//==============================================================================
template <typename T>
concept field_concept =
  has_real_type<T> &&
  has_num_dimensions<T> &&
  has_pos_type<T> &&
  has_tensor_type<T> &&
  requires(T f, typename T::pos_type x, typename T::real_type t,
           typename T::real_type tau) {
  { f(x, t) } -> std::convertible_to<typename T::pos_type>;
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
