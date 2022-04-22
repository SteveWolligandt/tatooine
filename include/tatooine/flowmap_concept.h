#ifndef TATOOINE_FLOWMAP_CONCEPT_H
#define TATOOINE_FLOWMAP_CONCEPT_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/vec.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename T, typename Real, std::size_t NumDimensions>
concept weak_flowmap_concept = requires(T f, vec<Real, NumDimensions> x, Real t, Real tau) {
  { f(x, t, tau) } ;
};
//==============================================================================
template <typename T>
concept flowmap_concept =
  has_real_type<T> &&
  has_num_dimensions<T> &&
  has_pos_type<T> &&
  requires(T f, typename T::pos_type x, typename T::real_type t,
           typename T::real_type tau) {
  { f(x, t, tau) } -> std::convertible_to<typename T::pos_type>;
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
