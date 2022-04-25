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
    requires(std::decay_t<T> f, typename std::decay_t<T>::pos_type x,
             typename std::decay_t<T>::real_type t) {
  { f(x, t) } -> std::same_as<typename T::tensor_type>;
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
