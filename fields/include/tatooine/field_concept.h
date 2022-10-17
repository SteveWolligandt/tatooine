#ifndef TATOOINE_FIELD_CONCEPT_H
#define TATOOINE_FIELD_CONCEPT_H
//==============================================================================
#include <tatooine/concepts.h>
#include <tatooine/vec.h>
#include <tatooine/field_type_traits.h>
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
    requires(std::decay_t<T> const f, field_pos_type<std::decay_t<T>> x,
             field_real_type< std::decay_t<T>> t) {
  { f(x, t) } -> std::same_as<field_tensor_type<std::decay_t<T>>>;
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
