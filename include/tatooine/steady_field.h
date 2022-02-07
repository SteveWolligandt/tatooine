#ifndef TATOOINE_FIXED_TIME_FIELD_H
#define TATOOINE_FIXED_TIME_FIELD_H
//==============================================================================
#include <tatooine/field.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Field>
struct steady_field
    : field<steady_field<Field>, typename Field::real_t,
            Field::num_dimensions(), typename Field::tensor_t> {
  using field_t  = Field;
  using this_t   = steady_field<Field>;
  using parent_type = field<this_t, typename Field::real_t,
                         Field::num_dimensions(), typename Field::tensor_t>;
  using typename parent_type::pos_t;
  using typename parent_type::real_t;
  using typename parent_type::tensor_t;
  //============================================================================
 private:
  Field m_internal_field;
  real_t  m_fixed_time;
  //============================================================================
 public:
  steady_field(steady_field const& other)     = default;
  steady_field(steady_field&& other) noexcept = default;

  auto operator=(steady_field const& other)
    -> steady_field& = default;
  auto operator=(steady_field&& other) noexcept 
    -> steady_field& = default;

  //============================================================================
  template <typename F_, typename TReal, enable_if<is_arithmetic<TReal>> = true>
  constexpr steady_field(F_&& f, TReal fixed_time)
      : m_internal_field{std::forward<F_>(f)},
        m_fixed_time{static_cast<real_t>(fixed_time)} {}
  //----------------------------------------------------------------------------
  constexpr auto evaluate(pos_t const& x, real_t const /*t*/) const -> tensor_t {
    return m_internal_field(x, m_fixed_time);
  }
  //----------------------------------------------------------------------------
  constexpr auto in_domain(pos_t const& x, real_t const /*t*/) const -> bool final {
    return m_internal_field.in_domain(x, m_fixed_time);
  }
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
