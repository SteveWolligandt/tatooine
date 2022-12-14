#ifndef TATOOINE_STEADY_FIELD_H
#define TATOOINE_STEADY_FIELD_H
//==============================================================================
#include <tatooine/field.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Field>
struct steady_field
    : field<steady_field<Field>, typename Field::real_type,
            Field::num_dimensions(), typename Field::tensor_type> {
  using field_t   = Field;
  using this_type = steady_field<Field>;
  using parent_type =
      field<this_type, typename Field::real_type, Field::num_dimensions(),
            typename Field::tensor_type>;
  using typename parent_type::pos_type;
  using typename parent_type::real_type;
  using typename parent_type::tensor_type;
  //============================================================================
 private:
  Field     m_internal_field;
  real_type m_fixed_time;
  //============================================================================
 public:
  steady_field(steady_field const& other)     = default;
  steady_field(steady_field&& other) noexcept = default;
  //----------------------------------------------------------------------------
  auto operator=(steady_field const& other) -> steady_field& = default;
  auto operator=(steady_field&& other) noexcept -> steady_field& = default;
  //----------------------------------------------------------------------------
  ~steady_field() = default;
  //============================================================================
  constexpr steady_field(convertible_to<Field> auto&& f, arithmetic auto fixed_time)
      : m_internal_field{std::forward<decltype(f)>(f)},
        m_fixed_time{static_cast<real_type>(fixed_time)} {}
  //----------------------------------------------------------------------------
  constexpr auto evaluate(pos_type const& x, real_type const /*t*/) const
      -> tensor_type {
    return m_internal_field(x, m_fixed_time);
  }
  //----------------------------------------------------------------------------
  constexpr auto internal() -> auto& { return m_internal_field; }
  constexpr auto internal() const -> auto const& { return m_internal_field; }
};
//------------------------------------------------------------------------------
template <typename F, typename T>
auto steady(F const& f, T const t) {
  return steady_field<F const&>{f, t};
}
//------------------------------------------------------------------------------
template <typename F, typename T>
auto steady(F&& f, T const t) {
  return steady_field<std::decay_t<F>>{std::forward<F>(f), t};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
