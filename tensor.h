#ifndef TATOOINE_TENSOR_H
#define TATOOINE_TENSOR_H

#include <array>
#include <iostream>
#include <ostream>
#include <random>
#include "crtp.h"
#include "functional.h"
#include "multiindexeddata.h"
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

template <typename tensor_t, typename real_t, size_t FixedDim, size_t... Dims>
struct tensor_slice;

//------------------------------------------------------------------------------
template <typename Tensor, typename Real, size_t... Dims>
struct base_tensor : crtp<Tensor> {
  using real_t   = Real;
  using tensor_t = Tensor;
  using this_t   = base_tensor<tensor_t, Real, Dims...>;
  using parent_t = crtp<tensor_t>;
  static constexpr auto   dimensions() { return std::array{Dims...}; }
  static constexpr auto   dimension(size_t i) { return dimensions()[i]; }
  static constexpr size_t num_dimensions() { return sizeof...(Dims); }
  static constexpr size_t num_components() { return (Dims * ...); }

  using parent_t::as_derived;
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

  //----------------------------------------------------------------------------
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
        tensor_t, real_t, FixedDim,
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
        const tensor_t, real_t, FixedDim,
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
    return static_multi_indexed_data<Dims...>::global_idx(is...);
  }


};

//==============================================================================
template <typename real_t, size_t... Dims>
struct tensor : base_tensor<tensor<real_t, Dims...>, real_t, Dims...> {
  //============================================================================
  using this_t   = tensor<real_t, Dims...>;
  using parent_t = base_tensor<this_t, real_t, Dims...>;
  using parent_t::parent_t;
  using parent_t::operator=;
  using parent_t::num_components;
  using parent_t::num_dimensions;

  //============================================================================
 private:
  std::array<real_t, num_components()> m_data;

  //============================================================================
 public:
  constexpr tensor() : m_data{make_array<real_t, num_components()>()} {}
  constexpr tensor(const tensor& other) = default;
  constexpr tensor(tensor&& other)      = default;
  constexpr tensor& operator=(const tensor& other) = default;
  constexpr tensor& operator=(tensor&& other) = default;

  //============================================================================
  constexpr tensor(real_t initial)
      : m_data{make_array<real_t, num_components()>(initial)} {}

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// vector initialization
  template <typename... Ts, size_t _n = num_dimensions(),
            std::enable_if_t<_n == 1>...>
  constexpr tensor(const Ts... ts) : m_data{static_cast<real_t>(ts)...} {
    static_assert(sizeof...(Ts) == parent_t::dimension(0));
  }

  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  /// matrix initialization
  template <typename... row_ts, size_t _n = num_dimensions(),
            std::enable_if_t<_n == 2>...>
  constexpr tensor(row_ts(&&... rows)[parent_t::dimension(1)])
      : m_data{make_array<real_t, num_components()>()} {
    static_assert(
        sizeof...(rows) == parent_t::dimension(0),
        "number of given rows does not match specified number of rows");

    // lambda inserting row into data block
    auto insert_row = [r = 0UL, this](const auto& row_data) mutable {
      size_t c = 0;
      for (auto v : row_data) { at(r, c++) = static_cast<real_t>(v); }
      ++r;
    };

    (insert_row(rows), ...);
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
  template <typename RandomEngine = std::mt19937_64>
  static constexpr auto rand(real_t min = 0, real_t max = 1,
                             RandomEngine&& eng = RandomEngine{
                                 std::random_device{}()}) {
    auto dist = [&]() {
      if constexpr (std::is_floating_point_v<real_t>) {
        return std::uniform_real_distribution<real_t>{min, max};
      } else if constexpr (std::is_integral_v<real_t>) {
        return std::uniform_int_distribution<real_t>{min, max};
      }
    }();
    this_t t;
    t.unary_operation([&](const auto& /*c*/) { return dist(eng); });
    return t;
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
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
template <size_t C, typename... Rows>
tensor(Rows const(&&... rows)[C])
    ->tensor<promote_t<Rows...>, sizeof...(Rows), C>;

//==============================================================================
template <typename real_t, size_t n>
struct vec : tensor<real_t, n> {
  using parent_t = tensor<real_t, n>;
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
template <typename real_t, size_t rows, size_t cols>
struct mat : tensor<real_t, rows, cols> {
  using parent_t = tensor<real_t, rows, cols>;
  using parent_t::parent_t;

  constexpr auto row(size_t i) { return this->template slice<0>(i); }
  constexpr auto col(size_t i) { return this->template slice<1>(i); }
  constexpr auto row(size_t i) const { return this->template slice<0>(i); }
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
template <typename Tensor, typename real_t, size_t FixedDim, size_t... Dims>
struct tensor_slice
    : base_tensor<tensor_slice<Tensor, real_t, FixedDim, Dims...>, real_t,
                  Dims...> {
  using tensor_t          = Tensor;
  using this_t            = tensor_slice<Tensor, real_t, FixedDim, Dims...>;
  using parent_t          = base_tensor<this_t, real_t, Dims...>;
  using parent_t::operator=;
  using parent_t::num_components;
  using parent_t::num_dimensions;

  //============================================================================
 private:
  tensor_t* m_tensor;
  size_t    m_fixed_index;

  //============================================================================
 public:
  constexpr tensor_slice(tensor_t* tensor, size_t fixed_index)
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
            typename _tensor_t = tensor_t,
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

template <typename tensor_t, typename real_t, size_t N>
constexpr auto norm(const base_tensor<tensor_t, real_t, N>& t_in, unsigned p = 2) {
  real_t n = 0;
  for (size_t i = 0; i < N; ++i) { n += std::pow(t_in(i), p); }
  return std::pow(n, real_t(1) / real_t(p));
}
//------------------------------------------------------------------------------
template <typename tensor_t, typename real_t, size_t N>
constexpr auto sqr_length(const base_tensor<tensor_t, real_t, N>& t_in) {
  real_t n = 0;
  for (size_t i = 0; i < N; ++i) { n += t_in(i) * t_in(i); }
  return n;
}
//------------------------------------------------------------------------------
template <typename tensor_t, typename real_t, size_t N>
constexpr auto length(const base_tensor<tensor_t, real_t, N>& t_in) {
  return std::sqrt(sqr_length(t_in));
}

//------------------------------------------------------------------------------
template <typename tensor_t, typename real_t, size_t N>
constexpr auto normalize(const base_tensor<tensor_t, real_t, N>& t_in) {
  return t_in / length(t_in);
}

//------------------------------------------------------------------------------
template <typename lhs_tensor_t, typename lhs_real_t, typename rhs_tensor_t,
          typename rhs_real_t, size_t N>
constexpr auto distance(const base_tensor<lhs_tensor_t, lhs_real_t, N>& lhs,
                        const base_tensor<rhs_tensor_t, rhs_real_t, N>& rhs) {
  return length(rhs - lhs);
}

//------------------------------------------------------------------------------
/// sum of all components of a vector
template <typename tensor_t, typename real_t, size_t VecDim>
constexpr auto sum(const base_tensor<tensor_t, real_t, VecDim>& v) {
  real_t s = 0;
  for (size_t i = 0; i < VecDim; ++i) { s += v(i); }
  return s;
}

//------------------------------------------------------------------------------
template <typename lhs_tensor_t, typename lhs_real_t, typename rhs_tensor_t,
          typename rhs_real_t, size_t N>
constexpr auto dot(const base_tensor<lhs_tensor_t, lhs_real_t, N>& lhs,
                   const base_tensor<rhs_tensor_t, rhs_real_t, N>& rhs) {
  promote_t<lhs_real_t, rhs_real_t> d = 0;
  for (size_t i = 0; i < N; ++i) { d += lhs(i) * rhs(i); }
  return d;
}

//------------------------------------------------------------------------------
template <typename tensor_t, typename real_t>
constexpr auto det(const base_tensor<tensor_t, real_t, 2, 2>& m) {
  return m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename tensor_t, typename real_t>
constexpr auto det(const base_tensor<tensor_t, real_t, 3, 3>& m) {
  return m(0, 0) * m(1, 1) * m(2, 2) + m(0, 1) * m(1, 2) * m(2, 0) +
         m(0, 2) * m(1, 0) * m(2, 1) - m(2, 0) * m(1, 1) * m(0, 2) -
         m(2, 1) * m(1, 2) * m(0, 0) - m(2, 2) * m(1, 0) * m(0, 2);
}

//------------------------------------------------------------------------------
template <typename lhs_tensor_t, typename lhs_real_t, typename rhs_tensor_t,
          typename rhs_real_t>
constexpr auto cross(const base_tensor<lhs_tensor_t, lhs_real_t, 3>& lhs,
                     const base_tensor<rhs_tensor_t, rhs_real_t, 3>& rhs) {
  return tensor<promote_t<lhs_real_t, rhs_real_t>, 3>{
      lhs(2) * rhs(3) - lhs(3) * lhs(2), lhs(3) * rhs(1) - lhs(1) * lhs(3),
      lhs(1) * rhs(2) - lhs(2) * lhs(1)};
}

//------------------------------------------------------------------------------
template <typename F, typename tensor_t, typename real_t, size_t... Dims>
constexpr auto unary_operation(
    F&& f, const base_tensor<tensor_t, real_t, Dims...>& t_in) {
  using out_real_t = typename std::result_of<decltype(f)(real_t)>::type;
  tensor<out_real_t, Dims...> t_out = t_in;
  return t_out.unary_operation(std::forward<F>(f));
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename F, typename lhs_tensor_t, typename lhs_real_t,
          typename rhs_tensor_t, typename rhs_real_t, size_t... Dims>
constexpr auto binary_operation(
    F&& f, const base_tensor<lhs_tensor_t, lhs_real_t, Dims...>& lhs,
    const base_tensor<rhs_tensor_t, rhs_real_t, Dims...>& rhs) {
  using out_real_t =
      typename std::result_of<decltype(f)(lhs_real_t, rhs_real_t)>::type;
  tensor<out_real_t, Dims...> t_out = lhs;
  return t_out.binary_operation(std::forward<F>(f), rhs);
}

//------------------------------------------------------------------------------
template <typename tensor_t, typename real_t, size_t... Dims>
constexpr auto operator-(const base_tensor<tensor_t, real_t, Dims...>& t) {
  tensor<real_t, Dims...> negated = t;
  return negated.unary_operation([](const auto& c) { return -c; });
}

//------------------------------------------------------------------------------
template <typename lhs_tensor_t, typename lhs_real_t, typename rhs_tensor_t,
          typename rhs_real_t, size_t... Dims>
constexpr auto operator+(
    const base_tensor<lhs_tensor_t, lhs_real_t, Dims...>& lhs,
    const base_tensor<rhs_tensor_t, rhs_real_t, Dims...>& rhs) {
  return binary_operation(std::plus<promote_t<lhs_real_t, rhs_real_t>>{}, lhs,
                          rhs);
}

//------------------------------------------------------------------------------
template <typename tensor_t, typename tensor_real_t, typename scalar_real_t,
          size_t... Dims,
          std::enable_if_t<std::is_arithmetic_v<scalar_real_t> ||
                           is_complex_v<scalar_real_t> ||
                           std::is_same_v<scalar_real_t, GiNaC::ex>>...>
constexpr auto operator*(const base_tensor<tensor_t, tensor_real_t, Dims...>& t,
                         const scalar_real_t scalar) {
  return unary_operation(
      [scalar](const auto& component) { return component * scalar; }, t);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename tensor_t, typename tensor_real_t, typename scalar_real_t,
          size_t... Dims,
          std::enable_if_t<std::is_arithmetic_v<scalar_real_t> ||
                           is_complex_v<scalar_real_t> ||
                           std::is_same_v<scalar_real_t, GiNaC::ex>>...>
constexpr auto operator*(
    const scalar_real_t                                  scalar,
    const base_tensor<tensor_t, tensor_real_t, Dims...>& t) {
  return unary_operation(
      [scalar](const auto& component) { return component * scalar; }, t);
}

//------------------------------------------------------------------------------
template <typename tensor_t, typename tensor_real_t, typename scalar_real_t,
          size_t... Dims,
          std::enable_if_t<std::is_arithmetic_v<scalar_real_t> ||
                           is_complex_v<scalar_real_t> ||
                           std::is_same_v<scalar_real_t, GiNaC::ex>>...>
constexpr auto operator/(const base_tensor<tensor_t, tensor_real_t, Dims...>& t,
                         const scalar_real_t scalar) {
  return unary_operation(
      [scalar](const auto& component) { return component / scalar; }, t);
}
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename tensor_t, typename tensor_real_t, typename scalar_real_t,
          size_t... Dims,
          std::enable_if_t<std::is_arithmetic_v<scalar_real_t> ||
                           is_complex_v<scalar_real_t> ||
                           std::is_same_v<scalar_real_t, GiNaC::ex>>...>
constexpr auto operator/(
    const scalar_real_t                                  scalar,
    const base_tensor<tensor_t, tensor_real_t, Dims...>& t) {
  return unary_operation(
      [scalar](const auto& component) { return scalar / component; }, t);
}

//------------------------------------------------------------------------------
template <typename lhs_tensor_t, typename lhs_real_t, typename rhs_tensor_t,
          typename rhs_real_t, size_t... Dims>
constexpr auto operator-(
    const base_tensor<lhs_tensor_t, lhs_real_t, Dims...>& lhs,
    const base_tensor<rhs_tensor_t, rhs_real_t, Dims...>& rhs) {
  return binary_operation(std::minus<promote_t<lhs_real_t, rhs_real_t>>{}, lhs,
                          rhs);
}

//------------------------------------------------------------------------------
/// matrix-vector-multiplication
template <typename lhs_tensor_t, typename lhs_real_t, typename rhs_tensor_t,
          typename rhs_real_t, size_t M, size_t N>
constexpr auto operator*(const base_tensor<lhs_tensor_t, lhs_real_t, M, N>& lhs,
                         const base_tensor<rhs_tensor_t, rhs_real_t, N>& rhs) {
  tensor<promote_t<lhs_real_t, rhs_real_t>, M> product;
  for (size_t i = 0; i < M; ++i) {
    product(i) = dot(lhs.template slice<0>(i), rhs);
  }
  return product;
}

//------------------------------------------------------------------------------
template <typename real_t, size_t m>
auto gesv(const tensor<real_t, m, m>& A, const tensor<real_t, m>& b) {
  auto           x = b;
  tensor<int, m> ipiv;
  int            nrhs = 1;
  if constexpr (std::is_same_v<float, real_t>) {
    LAPACKE_sgesv(LAPACK_COL_MAJOR, m, nrhs, const_cast<real_t*>(A.data()), m,
                  ipiv.data(), const_cast<real_t*>(x.data()), m);
  }
  if constexpr (std::is_same_v<double, real_t>) {
    LAPACKE_dgesv(LAPACK_COL_MAJOR, m, nrhs, const_cast<real_t*>(A.data()), m,
                  ipiv.data(), const_cast<real_t*>(x.data()), m);
  }
  return x;
}

//------------------------------------------------------------------------------
template <typename real_t, size_t m, size_t n>
auto gesv(const tensor<real_t, m, m>& A, const tensor<real_t, m, n>& B) {
  auto           X = B;
  tensor<int, m> ipiv;
  if constexpr (std::is_same_v<float, real_t>) {
    LAPACKE_sgesv(LAPACK_COL_MAJOR, m, n, const_cast<real_t*>(A.data()), m,
                  ipiv.data(), const_cast<real_t*>(X.data()), m);
  }

  if constexpr (std::is_same_v<double, real_t>) {
    LAPACKE_dgesv(LAPACK_COL_MAJOR, m, n, const_cast<real_t*>(A.data()), m,
                  ipiv.data(), const_cast<real_t*>(X.data()), m);
  }

  return X;
}






//------------------------------------------------------------------------------
template <typename real_t>
int eig_sort_compare(const real_t* lhs, const real_t* rhs) {
  return *lhs > *rhs;
}

//------------------------------------------------------------------------------
template <typename real_t, size_t m>
std::pair<mat<std::complex<real_t>, m, m>, vec<std::complex<real_t>, m>>
eigenvectors(tensor<real_t, m, m> A) {
  [[maybe_unused]] lapack_int info;
  std::array<real_t, m>       wr;
  std::array<real_t, m>       wi;
  std::array<real_t, m * m>   vr;
  if constexpr (std::is_same_v<double, real_t>) {
    info = LAPACKE_dgeev(LAPACK_COL_MAJOR, 'N', 'V', m, A.data(), m, wr.data(),
                         wi.data(), nullptr, m, vr.data(), m);
  }
  if constexpr (std::is_same_v<float, real_t>) {
    info = LAPACKE_sgeev(LAPACK_COL_MAJOR, 'N', 'V', m, A.data(), m, wr.data(),
                         wi.data(), nullptr, m, vr.data(), m);
  }

  vec<std::complex<real_t>, m> vals;
  mat<std::complex<real_t>, m, m> vecs;
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
template <typename real_t, size_t n>
auto eigenvalues_sym(tensor<real_t, n, n> A) {
  vec<real_t, n>              vals;
  [[maybe_unused]] lapack_int info;
  if constexpr (std::is_same_v<float, real_t>) {
    info = LAPACKE_ssyev(LAPACK_COL_MAJOR, 'N', 'U', n, A.data(), n,
                         vals.data());
  }
  if constexpr (std::is_same_v<double, real_t>) {
    info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'N', 'U', n, A.data(), n,
                         vals.data());
  }

  return vals;
}

//------------------------------------------------------------------------------
template <typename real_t, size_t n>
std::pair<mat<real_t, n, n>, vec<real_t, n>> eigenvectors_sym(
    mat<real_t, n, n> A) {
  vec<real_t, n>              vals;
  [[maybe_unused]] lapack_int info;
  if constexpr (std::is_same_v<float, real_t>) {
    info = LAPACKE_ssyev(LAPACK_COL_MAJOR, 'V', 'U', n, A.data(), n,
                         vals.data());
  }
  if constexpr (std::is_same_v<double, real_t>) {
    info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U', n, A.data(), n,
                         vals.data());
  }

  return {std::move(A), std::move(vals)};
}

////------------------------------------------------------------------------------
template <typename tensor_t, typename real_t, size_t... Dims>
auto real(const base_tensor<tensor_t, std::complex<real_t>, Dims...>& v) {
  tensor<real_t, Dims...> real_tensor;
  real_tensor.for_indices(
      [&](const auto... is) { real_tensor(is...) = v(is...).real(); });
  return real_tensor;
}

//------------------------------------------------------------------------------
template <typename tensor_t, typename real_t, size_t... Dims>
auto imag(const base_tensor<tensor_t, std::complex<real_t>, Dims...>& v) {
  tensor<real_t, Dims...> imag_tensor;
  imag_tensor.for_indices(
      [&](const auto... is) { imag_tensor(is...) = v(is...).imag(); });
  return imag_tensor;
}

//------------------------------------------------------------------------------
/// for comparison
template <typename lhs_tensor_t, typename lhs_real_t,
          typename rhs_tensor_t, typename rhs_real_t, size_t... Dims>
constexpr bool approx_equal(
    const base_tensor<lhs_tensor_t, lhs_real_t, Dims...>& lhs,
    const base_tensor<rhs_tensor_t, rhs_real_t, Dims...>& rhs,
    promote_t<lhs_real_t, rhs_real_t>            eps = 1e-6) {
  bool equal = true;
  lhs.for_indices([&](const auto... is) {
    if (std::abs(lhs(is...) - rhs(is...)) > eps) { equal = false; }
  });
  return equal;
}

//==============================================================================
// symbolic
//==============================================================================
template <typename out_real_t = double, typename tensor_t, size_t... Dims,
          typename... Relations>
auto evtod(const base_tensor<tensor_t, GiNaC::ex, Dims...>& t_in,
           Relations&&... relations) {
  tensor<out_real_t, Dims...> t_out;

  t_out.for_indices([&](const auto... is) {
    t_out(is...) = symbolic::evtod<out_real_t>(
        t_in(is...), std::forward<Relations>(relations)...);
  });

  return t_out;
}

//------------------------------------------------------------------------------
template <typename out_real_t = double, typename tensor_t, size_t... Dims,
          typename... Relations>
auto diff(const base_tensor<tensor_t, GiNaC::ex, Dims...>& t_in,
          const GiNaC::symbol& symbol, unsigned nth = 1) {
  return unary_operation(
      [&](const auto& component) { return component.diff(symbol, nth); }, t_in);
}

//==============================================================================
// I/O
//==============================================================================
/// printing vector
template <typename tensor_t, typename real_t, size_t N>
auto& operator<<(std::ostream& out, const base_tensor<tensor_t, real_t, N>& v) {
  out << "[ ";
  out << std::scientific;
  for (size_t i = 0; i < N; ++i) {
    if constexpr (!is_complex_v<real_t>) {
      if (v(i) >= 0) { out << ' '; }
    }
    out << v(i) << ' ';
  }
  out << "]";
  out << std::defaultfloat;
  return out;
}

template <typename tensor_t, typename real_t, size_t M, size_t N>
auto& operator<<(std::ostream&                              out,
                 const base_tensor<tensor_t, real_t, M, N>& m) {
  out << std::scientific;
  for (size_t j = 0; j < M; ++j) {
    out << "[ ";
    for (size_t i = 0; i < N; ++i) {
      if constexpr (!is_complex_v<real_t>) {
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
}  // namespace tatooine
//==============================================================================

#endif
