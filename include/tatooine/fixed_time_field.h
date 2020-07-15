#ifndef TATOOINE_FIXED_TIME_FIELD_H
#define TATOOINE_FIXED_TIME_FIELD_H
//==============================================================================
#include "field.h"
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename Field, typename Real, size_t N, size_t... TensorDims>
struct fixed_time_field : field<fixed_time_field<Field, Real, N, TensorDims...>,
                                Real, N, TensorDims...> {
  using field_t  = Field;
  using this_t   = fixed_time_field<Field, Real, N, TensorDims...>;
  using parent_t = field<this_t, Real, N, TensorDims...>;
  using typename parent_t::pos_t;
  using typename parent_t::tensor_t;
  //============================================================================
 private:
  Field const& m_field;
  Real         m_fixed_time;
  //============================================================================
 public:
  fixed_time_field(const fixed_time_field& other)     = default;
  fixed_time_field(fixed_time_field&& other) noexcept = default;
  fixed_time_field& operator=(const fixed_time_field& other) = default;
  fixed_time_field& operator=(fixed_time_field&& other) noexcept = default;

  //============================================================================
  template <typename TReal, enable_if_arithmetic<TReal> = true>
  constexpr fixed_time_field(const field<Field, Real, N, TensorDims...>& field,
                             TReal fixed_time)
      : m_field{field.as_derived()},
        m_fixed_time{static_cast<Real>(fixed_time)} {}
  //----------------------------------------------------------------------------
  constexpr tensor_t evaluate(const pos_t& x, Real /*t*/) const {
    return m_field(x, m_fixed_time);
  }
  //----------------------------------------------------------------------------
  constexpr auto in_domain(const pos_t& x, Real /*t*/) const -> bool final {
    return m_field.in_domain(x, m_fixed_time);
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
