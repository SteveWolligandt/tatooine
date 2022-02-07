#ifndef TATOOINE_CONSTANT_VECTORFIELD_H
#define TATOOINE_CONSTANT_VECTORFIELD_H
//==============================================================================
#include "field.h"
//==============================================================================
namespace tatooine {
//==============================================================================
/// constant vectorfield
template <typename Real, size_t N>
struct constant_vectorfield : field<constant_vectorfield<Real, N>, Real, N, N> {
  //============================================================================
  // typedefs
  //============================================================================
  using this_t   = constant_vectorfield<Real, N>;
  using parent_type = field<this_t, Real, N, N>;
  using typename parent_type::pos_t;
  using typename parent_type::tensor_t;
  //============================================================================
  // members
  //============================================================================
  const tensor_t m_vector;
  //============================================================================
  // ctors
  //============================================================================
  private:
   template <size_t... Is>
   constexpr constant_vectorfield(std::index_sequence<Is...> /*is*/)
       : m_vector{1, ((void)Is, 0)...} {}

  public:
   constexpr constant_vectorfield()
       : constant_vectorfield{std::make_index_sequence<N - 1>{}} {}
   //----------------------------------------------------------------------------
   constexpr constant_vectorfield(const vec<Real, N>& vector)
       : m_vector{vector} {}
   //----------------------------------------------------------------------------
   constexpr constant_vectorfield(vec<Real, N>&& vector)
       : m_vector{std::move(vector)} {}
   //============================================================================
   // methods
   //============================================================================
   constexpr tensor_t evaluate(const pos_t& /*x*/, Real /*t*/) const {
     return m_vector;
  }
  //----------------------------------------------------------------------------
  constexpr bool in_domain(const pos_t& /*x*/, Real /*t*/) const {
    return true;
  }
};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
