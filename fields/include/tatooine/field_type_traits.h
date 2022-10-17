#ifndef TATOOINE_FIELD_TYPE_TRAITS_H
#define TATOOINE_FIELD_TYPE_TRAITS_H
//==============================================================================
#include <tatooine/field.h>
//==============================================================================
namespace tatooine{
//==============================================================================
template <typename Field>
using field_real_type = typename std::decay_t<
    std::remove_pointer_t<std::decay_t<Field>>>::real_type;
//==============================================================================
template <typename Field>
using field_tensor_type = typename std::decay_t<
    std::remove_pointer_t<std::decay_t<Field>>>::tensor_type;
//==============================================================================
template <typename Field>
using field_pos_type = typename std::decay_t<
    std::remove_pointer_t<std::decay_t<Field>>>::pos_type;
//==============================================================================
template <typename Field>
static constexpr auto field_num_dimensions =
    std::decay_t<std::remove_pointer_t<std::decay_t<Field>>>::num_dimensions();
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
