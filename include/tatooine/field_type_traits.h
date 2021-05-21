#ifndef TATOOINE_FIELD_TYPE_TRAITS_H
#define TATOOINE_FIELD_TYPE_TRAITS_H
//==============================================================================
#include <tatooine/field.h>
//==============================================================================
namespace tatooine{
//==============================================================================
template <typename Field>
using field_real_t =
    typename std::decay_t<std::remove_pointer_t<Field>>::real_t;
//==============================================================================
template <typename Field>
using field_tensor_t =
    typename std::decay_t<std::remove_pointer_t<Field>>::tensor_t;
//==============================================================================
template <typename Field>
static constexpr auto field_num_dimensions =
    std::decay_t<std::remove_pointer_t<Field>>::num_dimensions();
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
