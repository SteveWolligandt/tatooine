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
concept flowmap_concept = has_real_type<T> && has_num_dimensions<T> &&
    has_pos_type<T> &&
    requires(std::decay_t<T> f, typename std::decay_t<T>::pos_type x,
             typename std::decay_t<T>::real_type t, typename std::decay_t<T>::real_type tau) {
  { f(x, t, tau) } -> std::same_as<typename std::decay_t<T>::pos_type>;
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
