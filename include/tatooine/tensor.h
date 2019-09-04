#ifndef TATOOINE_TENSOR_H
#define TATOOINE_TENSOR_H

#include <array>
#include <cassert>
#include <iostream>
#include <ostream>
#include "crtp.h"
#include "functional.h"
#include "multidimension.h"
#include "random.h"
#include "symbolic.h"
#include "type_traits.h"
#include "utility.h"

#include <lapacke.h>
#ifdef I
#undef I
#endif
//==============================================================================
namespace tatooine {
//==============================================================================

template <typename Real>
struct fill {
  Real value;
};
template <typename Real>
fill(Real) ->fill<Real>;

struct zeros_t {};
static constexpr inline zeros_t zeros;

struct ones_t {};
static constexpr inline ones_t ones;


template <typename Tensor, typename Real, size_t FixedDim, size_t... Dims>
struct tensor_slice;

//------------------------------------------------------------------------------
template <typename Tensor, typename Real, size_t... Dims>
struct base_tensor : crtp<Tensor> {
  using real_t   = Real;
  using tensor_t = Tensor;
  using this_t   = base_tensor<Tensor, Real, Dims...>;
  using parent_t = crtp<Tensor>;
  using parent_t::as_derived;

  //============================================================================
  static constexpr auto   dimensions() { return std::array{Dims...}; }
  static constexpr auto   dimension(size_t i) { return dimensions()[i]; }
  static constexpr size_t num_dimensions() { return sizeof...(Dims); }
  static constexpr size_t num_components() { return (Dims * ...); }
  static constexpr auto indices(){
    return multi_index{std::array{std::pair<size_t, size_t>{0, Dims - 1}...}};
  }
  template <typename F>
  static auto for_indices(F&& f) {
    for (auto is : indices()) {
      invoke_unpacked(std::forward<F>(f), unpack(is));
    }
  }

  //============================================================================
  constexpr base_tensor() = default;

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename other_tensor_t, typename other_real_t>
  constexpr base_tensor(
      const base_tensor<other_tensor_t, other_real_t, Dims...>& other) {
    assign_other_tensor(other);
  }

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename other_tensor_t, typename other_real_t>
  constexpr auto& operator=(
      const base_tensor<other_tensor_t, other_real_t, Dims...>& other) {
    assign_other_tensor(other);
    return *this;
  }

  //============================================================================
  template <typename F>
  auto& unary_operation(F&& f) {
    for_indices([this, &f](const auto... is) { at(is...) = f(at(is...)); });
    return as_derived();
  }

  //----------------------------------------------------------------------------
  template <typename F, typename other_tensor_t, typename other_real_t>
  decltype(auto) binary_operation(
      F&& f, const base_tensor<other_tensor_t, other_real_t, Dims...>& other) {
    for_indices([this, &f, &other](const auto... is) {
      at(is...) = f(at(is...), other(is...));
    });
    return as_derived();
  }

  //----------------------------------------------------------------------------
  template <typename other_tensor_t, typename other_real_t>
  constexpr void assign_other_tensor(
      const base_tensor<other_tensor_t, other_real_t, Dims...>& other) {
    for_indices([this, &other](const auto... is) { at(is...) = other(is...); });
  }

  //----------------------------------------------------------------------------
  template <typename... Is, enable_if_integral<Is...>...>
  constexpr const auto& at(const Is... is) const {
    static_assert(sizeof...(Is) == num_dimensions());
    return as_derived().at(is...);
  }

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Is, enable_if_integral<Is...>...>
  constexpr auto& at(const Is... is) {
    static_assert(sizeof...(Is) == num_dimensions());
    return as_derived().at(is...);
  }

  //----------------------------------------------------------------------------
  template <typename... Is, enable_if_integral<Is...>...>
  constexpr const auto& operator()(const Is... is) const {
    static_assert(sizeof...(Is) == num_dimensions());
    return at(is...);
  }

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Is, enable_if_integral<Is...>...>
  constexpr auto& operator()(const Is... is) {
    static_assert(sizeof...(Is) == num_dimensions());
    return at(is...);
  }

  //----------------------------------------------------------------------------
  template <size_t FixedDim, size_t... Is>
  constexpr auto slice(size_t fixed_index, std::index_sequence<Is...>) {
    static_assert(FixedDim < num_dimensions());
    return tensor_slice<
        Tensor, Real, FixedDim,
        dimension(sliced_indices<num_dimensions(), FixedDim>()[Is])...>{
        &as_derived(), fixed_index};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t FixedDim>
  constexpr auto slice(size_t fixed_index) {
    static_assert(FixedDim < num_dimensions());
    return slice<FixedDim>(fixed_index,
                           std::make_index_sequence<num_dimensions() - 1>{});
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t FixedDim, size_t... Is>
  constexpr auto slice(size_t fixed_index, std::index_sequence<Is...>) const {
    static_assert(FixedDim < num_dimensions());
    return tensor_slice<
        const Tensor, Real, FixedDim,
        dimension(sliced_indices<num_dimensions(), FixedDim>()[Is])...>{
        &as_derived(), fixed_index};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t FixedDim>
  constexpr auto slice(size_t fixed_index) const {
    static_assert(FixedDim < num_dimensions());
    return slice<FixedDim>(fixed_index,
                           std::make_index_sequence<num_dimensions() - 1>{});
  }

  //----------------------------------------------------------------------------
  template <typename... Is, enable_if_integral<Is...>...>
  static constexpr auto array_index(const Is... is) {
    static_assert(sizeof...(Is) == num_dimensions());
    return static_multidimension<Dims...>::global_idx(is...);
  }

  //----------------------------------------------------------------------------
  template <typename OtherTensor, typename OtherReal>
  auto& operator+=(const base_tensor<OtherTensor, OtherReal, Dims...>& other) {
    for_indices([&](const auto... is) { at(is...) += other(is...); });
    return *this;
  }

  //----------------------------------------------------------------------------
  template <typename OtherReal, enable_if_arithmetic_or_symbolic<OtherReal>...>
  auto& operator+=(const OtherReal& other) {
    for_indices([&](const auto... is) { at(is...) += other; });
    return *this;
  }

  //----------------------------------------------------------------------------
  template <typename OtherTensor, typename OtherReal>
  auto& operator-=(const base_tensor<OtherTensor, OtherReal, Dims...>& other) {
    for_indices([&](const auto... is) { at(is...) += other(is...); });
    return *this;
  }

  //----------------------------------------------------------------------------
  template <typename OtherReal, enable_if_arithmetic_or_symbolic<OtherReal>...>
  auto& operator-=(const OtherReal& other) {
    for_indices([&](const auto... is) { at(is...) -= other; });
    return *this;
  }

  //----------------------------------------------------------------------------
  template <typename OtherReal, enable_if_arithmetic_or_symbolic<OtherReal>...>
  auto& operator*=(const OtherReal& other) {
    for_indices([&](const auto... is) { at(is...) *= other; });
    return *this;
  }

  //----------------------------------------------------------------------------
  template <typename OtherReal, enable_if_arithmetic_or_symbolic<OtherReal>...>
  auto& operator/=(const OtherReal& other) {
    for_indices([&](const auto... is) { at(is...) /= other; });
    return *this;
  }
};

//==============================================================================
template <typename Real, size_t... Dims>
struct tensor : base_tensor<tensor<Real, Dims...>, Real, Dims...> {
  //============================================================================
  using this_t   = tensor<Real, Dims...>;
  using parent_t = base_tensor<this_t, Real, Dims...>;
  using parent_t::parent_t;
  using parent_t::operator=;
  using parent_t::num_components;
  using parent_t::num_dimensions;
  using parent_t::dimension;

  //============================================================================
 private:
  std::array<Real, num_components()> m_data;

  //============================================================================
 public:
  constexpr tensor() : m_data{make_array<Real, num_components()>()} {}
  constexpr tensor(const tensor& other)            = default;
  constexpr tensor(tensor&& other)                 = default;
  constexpr tensor& operator=(const tensor& other) = default;
  constexpr tensor& operator=(tensor&& other)      = default;
  ~tensor()                                        = default;

  //============================================================================
  public:
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// vector initialization
  template <typename... Ts, size_t _n = num_dimensions(),
            std::enable_if_t<_n == 1>...,
            enable_if_arithmetic_or_symbolic<Ts...>...>
  constexpr tensor(const Ts... ts) : m_data{static_cast<Real>(ts)...} {
    static_assert(sizeof...(Ts) == parent_t::dimension(0));
  }

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// matrix initialization
  template <typename... Rows, size_t _n = num_dimensions(),
            std::enable_if_t<_n == 2>...,
            enable_if_arithmetic_or_symbolic<Rows...>...>
  constexpr tensor(Rows(&&... rows)[parent_t::dimension(1)])
      : m_data{make_array<Real, num_components()>()} {
    static_assert(
        sizeof...(rows) == parent_t::dimension(0),
        "number of given rows does not match specified number of rows");

    // lambda inserting row into data block
    auto insert_row = [r = 0UL, this](const auto& row) mutable {
      for (size_t c = 0; c < dimension(1); ++c) {
        at(r, c) = static_cast<Real>(row[c]);
      }
      ++r;
    };

    (insert_row(rows), ...);
  }

  //----------------------------------------------------------------------------
  template <typename _real_t = Real, enable_if_arithmetic<_real_t>...>
  constexpr tensor(zeros_t /*zeros*/) : tensor{fill{0}} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename _real_t = Real, enable_if_arithmetic<_real_t>...>
  constexpr tensor(ones_t /*ones*/) : tensor{fill{1}} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename fill_real_t, typename _real_t = Real,
            enable_if_arithmetic<_real_t>...>
  constexpr tensor(fill<fill_real_t> f)
      : m_data{make_array<Real, num_components()>(f.value)} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename RandomReal, typename Engine, typename _real_t = Real,
            enable_if_arithmetic<RandomReal>...>
  constexpr tensor(random_uniform<RandomReal, Engine>&& rand) : tensor{} {
    this->unary_operation([&](const auto& /*c*/) { return rand.get(); });
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename RandomReal, typename Engine, typename _real_t = Real,
            enable_if_arithmetic<_real_t>...>
  constexpr tensor(random_normal<RandomReal, Engine>&& rand) : tensor{} {
    this->unary_operation([&](const auto& /*c*/) { return rand.get(); });
  }

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename other_tensor_t, typename other_real_t>
  constexpr tensor(
      const base_tensor<other_tensor_t, other_real_t, Dims...>& other) {
    this->assign_other_tensor(other);
  }

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename other_tensor_t, typename other_real_t>
  constexpr auto& operator=(
      const base_tensor<other_tensor_t, other_real_t, Dims...>& other) {
    this->assign_other_tensor(other);
    return *this;
  }

  //----------------------------------------------------------------------------
  static constexpr auto zeros() {
    return this_t{fill{0}};
  }

  //----------------------------------------------------------------------------
  static constexpr auto ones() {
    return this_t{fill{1}};
  }

  //----------------------------------------------------------------------------
  template <typename RandomEngine = std::mt19937_64>
  static constexpr auto randu(Real min = 0, Real max = 1,
                              RandomEngine&& eng = RandomEngine{
                                  std::random_device{}()}) {
    return this_t{random_uniform{eng, min, max}};
  }

  //----------------------------------------------------------------------------
  template <typename RandomEngine = std::mt19937_64>
  static constexpr auto randn(Real mean = 0, Real stddev = 1,
                              RandomEngine&& eng = RandomEngine{
                                  std::random_device{}()}) {
    return this_t{random_normal{eng, mean, stddev}};
  }

  //============================================================================
  template <typename... Is, enable_if_integral<Is...>...>
  constexpr const auto& at(const Is... is) const {
    static_assert(sizeof...(Is) == num_dimensions());
    return m_data[parent_t::array_index(is...)];
  }

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Is, enable_if_integral<Is...>...>
  constexpr auto& at(const Is... is) {
    static_assert(sizeof...(Is) == num_dimensions());
    return m_data[parent_t::array_index(is...)];
  }

  //----------------------------------------------------------------------------
  const auto& operator[](size_t i) const { return m_data[i]; }
  auto&       operator[](size_t i) { return m_data[i]; }

  //----------------------------------------------------------------------------
  decltype(auto) data() { return m_data.data(); }
  decltype(auto) data() const { return m_data.data(); }

  //----------------------------------------------------------------------------
  template <typename OtherReal>
  bool operator==(const tensor<OtherReal, Dims...>& other) const {
    return m_data == other.m_data;
  }

  //----------------------------------------------------------------------------
  template <typename OtherReal>
  bool operator<(const tensor<OtherReal, Dims...>& other) const {
    return m_data < other.m_data;
  }
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <size_t C, typename... Rows>
tensor(Rows const(&&... rows)[C])
    ->tensor<promote_t<Rows...>, sizeof...(Rows), C>;

//==============================================================================
template <typename Real, size_t n>
struct vec : tensor<Real, n> {
  using parent_t = tensor<Real, n>;
  using parent_t::parent_t;
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename... Ts>
vec(const Ts...)->vec<promote_t<Ts...>, sizeof...(Ts)>;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
using vec2 = vec<double, 2>;
using vec3 = vec<double, 3>;
using vec4 = vec<double, 4>;

//==============================================================================
template <typename Real, size_t rows, size_t cols>
struct mat : tensor<Real, rows, cols> {
  using parent_t = tensor<Real, rows, cols>;
  using parent_t::parent_t;

  constexpr auto row(size_t i) { return this->template slice<0>(i); }
  constexpr auto row(size_t i) const { return this->template slice<0>(i); }

  constexpr auto col(size_t i) { return this->template slice<1>(i); }
  constexpr auto col(size_t i) const { return this->template slice<1>(i); }
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <size_t C, typename... Rows>
mat(Rows const(&&... rows)[C])->mat<promote_t<Rows...>, sizeof...(Rows), C>;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
using mat2 = mat<double, 2, 2>;
using mat3 = mat<double, 3, 3>;
using mat4 = mat<double, 4, 4>;

//==============================================================================
template <typename Tensor, typename Real, size_t FixedDim, size_t... Dims>
struct tensor_slice
    : base_tensor<tensor_slice<Tensor, Real, FixedDim, Dims...>, Real,
                  Dims...> {
  using tensor_t          = Tensor;
  using this_t            = tensor_slice<Tensor, Real, FixedDim, Dims...>;
  using parent_t          = base_tensor<this_t, Real, Dims...>;
  using parent_t::operator=;
  using parent_t::num_components;
  using parent_t::num_dimensions;

  //============================================================================
 private:
  Tensor* m_tensor;
  size_t    m_fixed_index;

  //============================================================================
 public:
  constexpr tensor_slice(Tensor* tensor, size_t fixed_index)
      : m_tensor{tensor}, m_fixed_index{fixed_index} {}

  //----------------------------------------------------------------------------
  template <typename... Is, enable_if_integral<Is...>...>
  constexpr const auto& at(const Is... is) const {
    if constexpr (FixedDim == 0) {
      return m_tensor->at(m_fixed_index, is...);

    } else if constexpr (FixedDim == num_dimensions()) {
      return m_tensor->at(is..., m_fixed_index);

    } else {
      auto at_ = [this](const auto... is) -> decltype(auto) {
        return m_tensor->at(is...);
      };
      return invoke_unpacked(
          at_, unpack(extract<0, FixedDim - 1>(is...)), m_fixed_index,
          unpack(extract<FixedDim, num_dimensions() - 1>(is...)));
    };
  }

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename... Is, enable_if_integral<Is...>...,
            typename _tensor_t = Tensor,
            std::enable_if_t<!std::is_const_v<_tensor_t>>...>
  constexpr auto& at(const Is... is) {
    if constexpr (FixedDim == 0) {
      return m_tensor->at(m_fixed_index, is...);

    } else if constexpr (FixedDim == num_dimensions()) {
      return m_tensor->at(is..., m_fixed_index);

    } else {
      auto at_ = [this](const auto... is) -> decltype(auto) {
        return m_tensor->at(is...);
      };
      return invoke_unpacked(
          at_, unpack(extract<0, FixedDim - 1>(is...)), m_fixed_index,
          unpack(extract<FixedDim, num_dimensions() - 1>(is...)));
    };
  }
};

//==============================================================================
// operations
//==============================================================================

template <typename Tensor, typename Real, size_t N>
constexpr auto norm(const base_tensor<Tensor, Real, N>& t_in, unsigned p = 2) {
  Real n = 0;
  for (size_t i = 0; i < N; ++i) { n += std::pow(t_in(i), p); }
  return std::pow(n, Real(1) / Real(p));
}
//------------------------------------------------------------------------------
template <typename Tensor, typename Real, size_t N>
constexpr auto sqr_length(const base_tensor<Tensor, Real, N>& t_in) {
  Real n = 0;
  for (size_t i = 0; i < N; ++i) { n += t_in(i) * t_in(i); }
  return n;
}
//------------------------------------------------------------------------------
template <typename Tensor, typename Real, size_t N>
constexpr auto length(const base_tensor<Tensor, Real, N>& t_in) {
  return std::sqrt(sqr_length(t_in));
}

//------------------------------------------------------------------------------
template <typename Tensor, typename Real, size_t N>
constexpr auto normalize(const base_tensor<Tensor, Real, N>& t_in) {
  return t_in / length(t_in);
}

//------------------------------------------------------------------------------
template <typename LhsTensor, typename LhsReal, typename RhsTensor,
          typename RhsReal, size_t N>
constexpr auto distance(const base_tensor<LhsTensor, LhsReal, N>& lhs,
                        const base_tensor<RhsTensor, RhsReal, N>& rhs) {
  return length(rhs - lhs);
}

//------------------------------------------------------------------------------
/// sum of all components of a vector
template <typename Tensor, typename Real, size_t VecDim>
constexpr auto sum(const base_tensor<Tensor, Real, VecDim>& v) {
  Real s = 0;
  for (size_t i = 0; i < VecDim; ++i) { s += v(i); }
  return s;
}

//------------------------------------------------------------------------------
template <typename LhsTensor, typename LhsReal, typename RhsTensor,
          typename RhsReal, size_t N>
constexpr auto dot(const base_tensor<LhsTensor, LhsReal, N>& lhs,
                   const base_tensor<RhsTensor, RhsReal, N>& rhs) {
  promote_t<LhsReal, RhsReal> d = 0;
  for (size_t i = 0; i < N; ++i) { d += lhs(i) * rhs(i); }
  return d;
}

//------------------------------------------------------------------------------
template <typename Tensor, typename Real>
constexpr auto det(const base_tensor<Tensor, Real, 2, 2>& m) {
  return m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename Real>
constexpr auto det(const base_tensor<Tensor, Real, 3, 3>& m) {
  return m(0, 0) * m(1, 1) * m(2, 2) + m(0, 1) * m(1, 2) * m(2, 0) +
         m(0, 2) * m(1, 0) * m(2, 1) - m(2, 0) * m(1, 1) * m(0, 2) -
         m(2, 1) * m(1, 2) * m(0, 0) - m(2, 2) * m(1, 0) * m(0, 2);
}

//------------------------------------------------------------------------------
template <typename LhsTensor, typename LhsReal, typename RhsTensor,
          typename RhsReal>
constexpr auto cross(const base_tensor<LhsTensor, LhsReal, 3>& lhs,
                     const base_tensor<RhsTensor, RhsReal, 3>& rhs) {
  return tensor<promote_t<LhsReal, RhsReal>, 3>{
      lhs(2) * rhs(3) - lhs(3) * lhs(2), lhs(3) * rhs(1) - lhs(1) * lhs(3),
      lhs(1) * rhs(2) - lhs(2) * lhs(1)};
}

//------------------------------------------------------------------------------
template <typename F, typename Tensor, typename Real, size_t... Dims>
constexpr auto unary_operation(
    F&& f, const base_tensor<Tensor, Real, Dims...>& t_in) {
  using RealOut = typename std::result_of<decltype(f)(Real)>::type;
  tensor<RealOut, Dims...> t_out = t_in;
  return t_out.unary_operation(std::forward<F>(f));
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename F, typename LhsTensor, typename LhsReal,
          typename RhsTensor, typename RhsReal, size_t... Dims>
constexpr auto binary_operation(
    F&& f, const base_tensor<LhsTensor, LhsReal, Dims...>& lhs,
    const base_tensor<RhsTensor, RhsReal, Dims...>& rhs) {
  using RealOut =
      typename std::result_of<decltype(f)(LhsReal, RhsReal)>::type;
  tensor<RealOut, Dims...> t_out = lhs;
  return t_out.binary_operation(std::forward<F>(f), rhs);
}

//------------------------------------------------------------------------------
template <typename Tensor, typename Real, size_t... Dims>
constexpr auto operator-(const base_tensor<Tensor, Real, Dims...>& t) {
  tensor<Real, Dims...> negated = t;
  return negated.unary_operation([](const auto& c) { return -c; });
}

//------------------------------------------------------------------------------
template <typename LhsTensor, typename LhsReal, typename RhsReal,
          size_t... Dims, enable_if_arithmetic<RhsReal>...>
constexpr auto operator+(const base_tensor<LhsTensor, LhsReal, Dims...>& lhs,
                         RhsReal scalar) {
  return unary_operation([scalar](const auto& c) { return c + scalar; }, lhs);
}

//------------------------------------------------------------------------------
template <typename LhsTensor, typename LhsReal, typename RhsTensor,
          typename RhsReal, size_t M, size_t N, size_t O>
constexpr auto operator*(
    const base_tensor<LhsTensor, LhsReal, M, N>& lhs,
    const base_tensor<RhsTensor, RhsReal, N, O>& rhs) {
  mat<promote_t<LhsReal, RhsReal>, M, O> product;
  for (size_t r = 0; r < M; ++r) {
    for (size_t c = 0; c < O; ++c) {
      product(r, c) = dot(lhs.template slice<0>(r), rhs.template slice<1>(c));
    }
  }
  return product;
}

//------------------------------------------------------------------------------
template <typename LhsTensor, typename LhsReal, typename RhsTensor,
          typename RhsReal, size_t... Dims,
          std::enable_if_t<(sizeof...(Dims) != 2)>...>
constexpr auto operator*(const base_tensor<LhsTensor, LhsReal, Dims...>& lhs,
                         const base_tensor<RhsTensor, RhsReal, Dims...>& rhs) {
  return binary_operation(std::multiplies<promote_t<LhsReal, RhsReal>>{}, lhs,
                          rhs);
}

//------------------------------------------------------------------------------
template <typename LhsTensor, typename LhsReal, typename RhsTensor,
          typename RhsReal, size_t... Dims>
constexpr auto operator/(
    const base_tensor<LhsTensor, LhsReal, Dims...>& lhs,
    const base_tensor<RhsTensor, RhsReal, Dims...>& rhs) {
  return binary_operation(std::divides<promote_t<LhsReal, RhsReal>>{}, lhs,
                          rhs);
}

//------------------------------------------------------------------------------
template <typename LhsTensor, typename LhsReal, typename RhsTensor,
          typename RhsReal, size_t... Dims>
constexpr auto operator+(
    const base_tensor<LhsTensor, LhsReal, Dims...>& lhs,
    const base_tensor<RhsTensor, RhsReal, Dims...>& rhs) {
  return binary_operation(std::plus<promote_t<LhsReal, RhsReal>>{}, lhs,
                          rhs);
}

//------------------------------------------------------------------------------
template <typename Tensor, typename tensor_real_t, typename scalar_real_t,
          size_t... Dims,
          std::enable_if_t<std::is_arithmetic_v<scalar_real_t> ||
                           is_complex_v<scalar_real_t> ||
                           std::is_same_v<scalar_real_t, GiNaC::ex>>...>
constexpr auto operator*(const base_tensor<Tensor, tensor_real_t, Dims...>& t,
                         const scalar_real_t scalar) {
  return unary_operation(
      [scalar](const auto& component) { return component * scalar; }, t);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename tensor_real_t, typename scalar_real_t,
          size_t... Dims,
          std::enable_if_t<std::is_arithmetic_v<scalar_real_t> ||
                           is_complex_v<scalar_real_t> ||
                           std::is_same_v<scalar_real_t, GiNaC::ex>>...>
constexpr auto operator*(
    const scalar_real_t                                  scalar,
    const base_tensor<Tensor, tensor_real_t, Dims...>& t) {
  return unary_operation(
      [scalar](const auto& component) { return component * scalar; }, t);
}

//------------------------------------------------------------------------------
template <typename Tensor, typename tensor_real_t, typename scalar_real_t,
          size_t... Dims,
          std::enable_if_t<std::is_arithmetic_v<scalar_real_t> ||
                           is_complex_v<scalar_real_t> ||
                           std::is_same_v<scalar_real_t, GiNaC::ex>>...>
constexpr auto operator/(const base_tensor<Tensor, tensor_real_t, Dims...>& t,
                         const scalar_real_t scalar) {
  return unary_operation(
      [scalar](const auto& component) { return component / scalar; }, t);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename tensor_real_t, typename scalar_real_t,
          size_t... Dims,
          std::enable_if_t<std::is_arithmetic_v<scalar_real_t> ||
                           is_complex_v<scalar_real_t> ||
                           std::is_same_v<scalar_real_t, GiNaC::ex>>...>
constexpr auto operator/(
    const scalar_real_t                                  scalar,
    const base_tensor<Tensor, tensor_real_t, Dims...>& t) {
  return unary_operation(
      [scalar](const auto& component) { return scalar / component; }, t);
}

//------------------------------------------------------------------------------
template <typename LhsTensor, typename LhsReal, typename RhsTensor,
          typename RhsReal, size_t... Dims>
constexpr auto operator-(
    const base_tensor<LhsTensor, LhsReal, Dims...>& lhs,
    const base_tensor<RhsTensor, RhsReal, Dims...>& rhs) {
  return binary_operation(std::minus<promote_t<LhsReal, RhsReal>>{}, lhs,
                          rhs);
}

//------------------------------------------------------------------------------
/// matrix-vector-multiplication
template <typename LhsTensor, typename LhsReal, typename RhsTensor,
          typename RhsReal, size_t M, size_t N>
constexpr auto operator*(const base_tensor<LhsTensor, LhsReal, M, N>& lhs,
                         const base_tensor<RhsTensor, RhsReal, N>& rhs) {
  tensor<promote_t<LhsReal, RhsReal>, M> product;
  for (size_t i = 0; i < M; ++i) {
    product(i) = dot(lhs.template slice<0>(i), rhs);
  }
  return product;
}

//------------------------------------------------------------------------------
template <typename Real, size_t m>
auto gesv(const tensor<Real, m, m>& A, const tensor<Real, m>& b) {
  auto           x = b;
  tensor<int, m> ipiv;
  int            nrhs = 1;
  if constexpr (std::is_same_v<float, Real>) {
    LAPACKE_sgesv(LAPACK_COL_MAJOR, m, nrhs, const_cast<Real*>(A.data()), m,
                  ipiv.data(), const_cast<Real*>(x.data()), m);
  }
  if constexpr (std::is_same_v<double, Real>) {
    LAPACKE_dgesv(LAPACK_COL_MAJOR, m, nrhs, const_cast<Real*>(A.data()), m,
                  ipiv.data(), const_cast<Real*>(x.data()), m);
  }
  return x;
}

//------------------------------------------------------------------------------
template <typename Real, size_t m, size_t n>
auto gesv(const tensor<Real, m, m>& A, const tensor<Real, m, n>& B) {
  auto           X = B;
  tensor<int, m> ipiv;
  if constexpr (std::is_same_v<float, Real>) {
    LAPACKE_sgesv(LAPACK_COL_MAJOR, m, n, const_cast<Real*>(A.data()), m,
                  ipiv.data(), const_cast<Real*>(X.data()), m);
  }

  if constexpr (std::is_same_v<double, Real>) {
    LAPACKE_dgesv(LAPACK_COL_MAJOR, m, n, const_cast<Real*>(A.data()), m,
                  ipiv.data(), const_cast<Real*>(X.data()), m);
  }

  return X;
}






//------------------------------------------------------------------------------
template <typename Real>
int eig_sort_compare(const Real* lhs, const Real* rhs) {
  return *lhs > *rhs;
}

//------------------------------------------------------------------------------
template <typename Real, size_t m>
std::pair<mat<std::complex<Real>, m, m>, vec<std::complex<Real>, m>>
eigenvectors(tensor<Real, m, m> A) {
  [[maybe_unused]] lapack_int info;
  std::array<Real, m>       wr;
  std::array<Real, m>       wi;
  std::array<Real, m * m>   vr;
  if constexpr (std::is_same_v<double, Real>) {
    info = LAPACKE_dgeev(LAPACK_COL_MAJOR, 'N', 'V', m, A.data(), m, wr.data(),
                         wi.data(), nullptr, m, vr.data(), m);
  }
  if constexpr (std::is_same_v<float, Real>) {
    info = LAPACKE_sgeev(LAPACK_COL_MAJOR, 'N', 'V', m, A.data(), m, wr.data(),
                         wi.data(), nullptr, m, vr.data(), m);
  }

  vec<std::complex<Real>, m> vals;
  mat<std::complex<Real>, m, m> vecs;
  for (size_t i = 0; i < m; ++i) { vals[i] = {wr[i], wi[i]}; }
  for (size_t j = 0; j < m; ++j) {
    for (size_t i = 0; i < m; ++i) {
      if (wi[j] == 0) {
        vecs(i, j) = {vr[i + j * m], 0};
      } else {
        vecs(i, j)     = {vr[i + j * m], vr[i + (j + 1) * m]};
        vecs(i, j + 1) = {vr[i + j * m], -vr[i + (j + 1) * m]};
        if (i == m - 1) { ++j; }
      }
    }
  }

  return {std::move(vecs), std::move(vals)};
}

//------------------------------------------------------------------------------
template <typename Real, size_t n>
auto eigenvalues_sym(tensor<Real, n, n> A) {
  vec<Real, n>              vals;
  [[maybe_unused]] lapack_int info;
  if constexpr (std::is_same_v<float, Real>) {
    info = LAPACKE_ssyev(LAPACK_COL_MAJOR, 'N', 'U', n, A.data(), n,
                         vals.data());
  }
  if constexpr (std::is_same_v<double, Real>) {
    info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'N', 'U', n, A.data(), n,
                         vals.data());
  }

  return vals;
}

//------------------------------------------------------------------------------
template <typename Real, size_t n>
std::pair<mat<Real, n, n>, vec<Real, n>> eigenvectors_sym(
    mat<Real, n, n> A) {
  vec<Real, n>              vals;
  [[maybe_unused]] lapack_int info;
  if constexpr (std::is_same_v<float, Real>) {
    info = LAPACKE_ssyev(LAPACK_COL_MAJOR, 'V', 'U', n, A.data(), n,
                         vals.data());
  }
  if constexpr (std::is_same_v<double, Real>) {
    info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U', n, A.data(), n,
                         vals.data());
  }

  return {std::move(A), std::move(vals)};
}

////------------------------------------------------------------------------------
template <typename Tensor, typename Real, size_t... Dims>
auto real(const base_tensor<Tensor, std::complex<Real>, Dims...>& v) {
  tensor<Real, Dims...> real_tensor;
  real_tensor.for_indices(
      [&](const auto... is) { real_tensor(is...) = v(is...).real(); });
  return real_tensor;
}

//------------------------------------------------------------------------------
template <typename Tensor, typename Real, size_t... Dims>
auto imag(const base_tensor<Tensor, std::complex<Real>, Dims...>& v) {
  tensor<Real, Dims...> imag_tensor;
  imag_tensor.for_indices(
      [&](const auto... is) { imag_tensor(is...) = v(is...).imag(); });
  return imag_tensor;
}

//------------------------------------------------------------------------------
/// for comparison
template <typename LhsTensor, typename LhsReal,
          typename RhsTensor, typename RhsReal, size_t... Dims>
constexpr bool approx_equal(
    const base_tensor<LhsTensor, LhsReal, Dims...>& lhs,
    const base_tensor<RhsTensor, RhsReal, Dims...>& rhs,
    promote_t<LhsReal, RhsReal>            eps = 1e-6) {
  bool equal = true;
  lhs.for_indices([&](const auto... is) {
    if (std::abs(lhs(is...) - rhs(is...)) > eps) { equal = false; }
  });
  return equal;
}

//==============================================================================
// symbolic
//==============================================================================
template <typename RealOut = double, typename Tensor, size_t... Dims,
          typename... Relations>
auto evtod(const base_tensor<Tensor, GiNaC::ex, Dims...>& t_in,
           Relations&&... relations) {
  tensor<RealOut, Dims...> t_out;

  t_out.for_indices([&](const auto... is) {
    t_out(is...) = symbolic::evtod<RealOut>(
        t_in(is...), std::forward<Relations>(relations)...);
  });

  return t_out;
}

//------------------------------------------------------------------------------
template <typename RealOut = double, typename Tensor, size_t... Dims,
          typename... Relations>
auto diff(const base_tensor<Tensor, GiNaC::ex, Dims...>& t_in,
          const GiNaC::symbol& symbol, unsigned nth = 1) {
  return unary_operation(
      [&](const auto& component) { return component.diff(symbol, nth); }, t_in);
}

//------------------------------------------------------------------------------
template <typename Tensor, size_t M, size_t N>
auto to_ginac_matrix(const base_tensor<Tensor, GiNaC::ex, M, N>& m_in) {
  GiNaC::matrix m_out(M, N);
  m_in.for_indices([&](const auto... is) { m_out(is...) = m_in(is...); });
  return m_out;
}

//------------------------------------------------------------------------------
template <size_t M, size_t N>
auto to_mat(const GiNaC::matrix& m_in) {
  assert(m_in.rows() == M);
  assert(m_in.cols() == N);
  mat<GiNaC::ex, M, N> m_out;
  m_out.for_indices([&](const auto... is) { m_out(is...) = m_in(is...); });
  return m_out;
}

//------------------------------------------------------------------------------
template <typename Tensor, size_t... Dims>
void eval(base_tensor<Tensor, GiNaC::ex, Dims...>& m) {
  m.for_indices([&m](const auto... is) { m(is...) = m(is...).eval(); });
}
//------------------------------------------------------------------------------
template <typename Tensor, size_t... Dims>
void evalf(base_tensor<Tensor, GiNaC::ex, Dims...>& m) {
  m.for_indices([&m](const auto... is) { m(is...) = m(is...).evalf(); });
}
//------------------------------------------------------------------------------
template <typename Tensor, size_t... Dims>
void evalm(base_tensor<Tensor, GiNaC::ex, Dims...>& m) {
  m.for_indices([&m](const auto... is) { m(is...) = m(is...).evalm(); });
}

//------------------------------------------------------------------------------
template <typename Tensor, size_t... Dims>
void expand(base_tensor<Tensor, GiNaC::ex, Dims...>& m) {
  m.for_indices([&m](const auto... is) { m(is...) = m(is...).expand(); });
}

//------------------------------------------------------------------------------
template <typename Tensor, size_t... Dims>
void normal(base_tensor<Tensor, GiNaC::ex, Dims...>& m) {
  m.for_indices([&m](const auto... is) { m(is...).normal(); });
}

//------------------------------------------------------------------------------
template <typename Tensor, size_t M, size_t N>
auto inverse(const base_tensor<Tensor, GiNaC::ex, M, N>& m_in) {
  return to_mat<M, N>(to_ginac_matrix(m_in).inverse());
}

//==============================================================================
// I/O
//==============================================================================
/// printing vector
template <typename Tensor, typename Real, size_t N>
auto& operator<<(std::ostream& out, const base_tensor<Tensor, Real, N>& v) {
  out << "[ ";
  out << std::scientific;
  for (size_t i = 0; i < N; ++i) {
    if constexpr (!is_complex_v<Real>) {
      if (v(i) >= 0) { out << ' '; }
    }
    out << v(i) << ' ';
  }
  out << "]";
  out << std::defaultfloat;
  return out;
}

template <typename Tensor, typename Real, size_t M, size_t N>
auto& operator<<(std::ostream&                              out,
                 const base_tensor<Tensor, Real, M, N>& m) {
  out << std::scientific;
  for (size_t j = 0; j < M; ++j) {
    out << "[ ";
    for (size_t i = 0; i < N; ++i) {
      if constexpr (!is_complex_v<Real>) {
        if (m(j, i) >= 0) { out << ' '; }
      }
      out << m(j, i) << ' ';
    }
    out << "]\n";
  }
  out << std::defaultfloat;
  return out;
}

//==============================================================================
// type traits
//==============================================================================
template <typename Tensor, typename Real, size_t n>
struct num_components<base_tensor<Tensor, Real, n>>
    : std::integral_constant<size_t, n> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t n>
struct num_components<tensor<Real, n>> : std::integral_constant<size_t, n> {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t n>
struct num_components<vec<Real, n>> : std::integral_constant<size_t, n> {};

//==============================================================================
template <typename Real, size_t N>
struct internal_data_type<vec<Real, N>> {
  using type = Real;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t M, size_t N>
struct internal_data_type<mat<Real, M, N>> {
  using type = Real;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real, size_t... Dims>
struct internal_data_type<tensor<Real, Dims...>> {
  using type = Real;
};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Tensor, typename Real, size_t... Dims>
struct internal_data_type<base_tensor<Tensor, Real, Dims...>> {
  using type = Real;
};

//==============================================================================
template <typename Tensor, typename Real, size_t N>
struct unpack<base_tensor<Tensor, Real, N>> {
  static constexpr size_t n = N;
  base_tensor<Tensor, Real, N>&       container;

  //----------------------------------------------------------------------------
  constexpr unpack(base_tensor<Tensor, Real, N>& c) : container{c} {}

  //----------------------------------------------------------------------------
  template <size_t I>
  constexpr auto& get() {
    return container(I);
  }

  //----------------------------------------------------------------------------
  template <size_t I>
  constexpr const auto& get() const {
    return container(I);
  }
};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename Tensor, typename Real, size_t N>
unpack(base_tensor<Tensor, Real, N>& c)->unpack<base_tensor<Tensor, Real, N>>;

//==============================================================================
template <typename Tensor, typename Real, size_t N>
struct unpack<const base_tensor<Tensor, Real, N>> {
  static constexpr size_t n = N;
  const base_tensor<Tensor, Real, N>& container;

  //----------------------------------------------------------------------------
  constexpr unpack(const base_tensor<Tensor, Real, N>& c) : container{c} {}

  //----------------------------------------------------------------------------
  template <size_t I>
  constexpr const auto& get() const {
    return container(I);
  }
};
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <typename Tensor, typename Real, size_t N>
unpack(const base_tensor<Tensor, Real, N>& c)
    ->unpack<const base_tensor<Tensor, Real, N>>;

//==============================================================================
template <typename Real, size_t N>
struct unpack<tensor<Real, N>> {
  static constexpr size_t n = N;
  tensor<Real, N>&       container;

  //----------------------------------------------------------------------------
  constexpr unpack(tensor<Real, N>& c) : container{c} {}

  //----------------------------------------------------------------------------
  template <size_t I>
  constexpr auto& get() {
    return container[I];
  }

  //----------------------------------------------------------------------------
  template <size_t I>
  constexpr const auto& get() const {
    return container[I];
  }
};
//==============================================================================
template <typename Real, size_t N>
unpack(tensor<Real, N>& c)->unpack<tensor<Real, N>>;

//==============================================================================
template <typename Real, size_t N>
struct unpack<const tensor<Real, N>> {
  static constexpr size_t n = N;
  const tensor<Real, N>& container;

  //----------------------------------------------------------------------------
  constexpr unpack(const tensor<Real, N>& c) : container{c} {}

  //----------------------------------------------------------------------------
  template <size_t I>
  constexpr const auto& get() const {
    return container[I];
  }
};
//==============================================================================
template <typename Real, size_t N>
unpack(const tensor<Real, N>& c)->unpack<const tensor<Real, N>>;

//==============================================================================
}  // namespace tatooine
//==============================================================================

#endif
