#ifndef TATOOINE_DYNAMIC_TENSOR_H
#define TATOOINE_DYNAMIC_TENSOR_H
//==============================================================================
#include <tatooine/multidim_array.h>
#include <sstream>
#include <ostream>
//==============================================================================
namespace tatooine{
//==============================================================================
template <typename T>
struct is_dynamic_tensor : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
static constexpr auto is_dynamic_tensor_v = is_dynamic_tensor<T>::value;
//==============================================================================
template <real_or_complex_number T>
struct dynamic_tensor : dynamic_multidim_array<T> {
  using this_t   = dynamic_tensor<T>;
  using parent_t = dynamic_multidim_array<T>;
  using parent_t::parent_t;
  //============================================================================
  // factories
  //============================================================================
  static auto zeros(integral auto... size) {
    return this_t{tag::zeros, size...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // -
  template <unsigned_integral UInt>
  static auto zeros(std::vector<UInt> const& size) {
    return this_t{tag::zeros, size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // -
  template <unsigned_integral UInt, size_t N>
  static auto zeros(std::array<UInt, N> const& size) {
    return this_t{tag::zeros, size};
  }
  //------------------------------------------------------------------------------
  static auto ones(integral auto... size) { return this_t{tag::ones, size...}; }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // -
  template <unsigned_integral UInt>
  static auto ones(std::vector<UInt> const& size) {
    return this_t{tag::ones, size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // -
  template <size_t N, unsigned_integral UInt>
  static auto ones(std::array<UInt, N> const& size) {
    return this_t{tag::ones, size};
  }
  //------------------------------------------------------------------------------
  // template <unsigned_integral UInt, typename RandEng = std::mt19937_64>
  // static auto randu(T min, T max, std::initializer_list<UInt>&& size,
  //                  RandEng&& eng = RandEng{std::random_device{}()}) {
  //  return this_t{random_uniform{min, max, std::forward<RandEng>(eng)},
  //                std::vector<UInt>(std::move(size))};
  //}
  //// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  ///-
  ///-
  // template <unsigned_integral UInt, typename RandEng = std::mt19937_64>
  // static auto randu(std::initializer_list<UInt>&& size, T min = 0, T
  // max = 1,
  //                  RandEng&& eng = RandEng{std::random_device{}()}) {
  //  return this_t{random_uniform{min, max, std::forward<RandEng>(eng)},
  //                std::vector<UInt>(std::move(size))};
  //}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // -
  template <unsigned_integral UInt, typename RandEng = std::mt19937_64>
  static auto randu(T min, T max, std::vector<UInt> const& size,
                    RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_t{
        random_uniform<T, RandEng>{min, max, std::forward<RandEng>(eng)}, size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // -
  template <unsigned_integral UInt, typename RandEng = std::mt19937_64>
  static auto randu(std::vector<UInt> const& size, T min = 0, T max = 1,
                    RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_t{
        random_uniform<T, RandEng>{min, max, std::forward<RandEng>(eng)}, size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // -
  template <size_t N, unsigned_integral UInt,
            typename RandEng = std::mt19937_64>
  static auto randu(T min, T max, std::array<UInt, N> const& size,
                    RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_t{
        random_uniform<T, RandEng>{min, max, std::forward<RandEng>(eng)}, size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // -
  template <size_t N, unsigned_integral UInt,
            typename RandEng = std::mt19937_64>
  static auto randu(std::array<UInt, N> const& size, T min = 0, T max = 1,
                    RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_t{
        random_uniform<T, RandEng>{min, max, std::forward<RandEng>(eng)}, size};
  }
  //----------------------------------------------------------------------------
  template <unsigned_integral UInt, typename RandEng>
  static auto rand(random_uniform<T, RandEng> const& rand,
                   std::vector<UInt> const&          size) {
    return this_t{rand, size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // -
  template <size_t N, unsigned_integral UInt, typename RandEng>
  static auto rand(random_uniform<T, RandEng> const& rand,
                   std::array<UInt, N> const&        size) {
    return this_t{rand, size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // -
  template <typename RandEng>
  static auto rand(random_uniform<T, RandEng> const& rand,
                   integral auto... size) {
    return this_t{rand, std::vector{static_cast<size_t>(size)...}};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // -
  template <unsigned_integral UInt, typename RandEng>
  static auto rand(random_uniform<T, RandEng>&& rand,
                   std::vector<UInt> const&     size) {
    return this_t{std::move(rand), size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // -
  template <size_t N, unsigned_integral UInt, typename RandEng>
  static auto rand(random_uniform<T, RandEng>&& rand,
                   std::array<UInt, N> const&   size) {
    return this_t{std::move(rand), size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // -
  template <typename RandEng>
  static auto rand(random_uniform<T, RandEng>&& rand, integral auto... size) {
    return this_t{std::move(rand), std::vector{static_cast<size_t>(size)...}};
  }
  //----------------------------------------------------------------------------
  template <unsigned_integral UInt, typename RandEng>
  static auto rand(random_normal<T, RandEng> const& rand,
                   std::vector<UInt> const&         size) {
    return this_t{rand, size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // -
  template <size_t N, unsigned_integral UInt, typename RandEng>
  static auto rand(random_normal<T, RandEng> const& rand,
                   std::array<UInt, N> const&       size) {
    return this_t{rand, size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // -
  template <typename RandEng>
  static auto rand(random_normal<T, RandEng> const& rand,
                   integral auto... size) {
    return this_t{rand, std::vector{static_cast<size_t>(size)...}};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // -
  template <unsigned_integral UInt, typename RandEng>
  static auto rand(random_normal<T, RandEng>&& rand,
                   std::vector<UInt> const&    size) {
    return this_t{std::move(rand), size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // -
  template <size_t N, unsigned_integral UInt, typename RandEng>
  static auto rand(random_normal<T, RandEng>&& rand,
                   std::array<UInt, N> const&  size) {
    return this_t{std::move(rand), size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // -
  template <typename RandEng>
  static auto rand(random_normal<T, RandEng>&& rand, integral auto... size) {
    return this_t{std::move(rand), std::vector{static_cast<size_t>(size)...}};
  }
};
//------------------------------------------------------------------------------
template <typename T>
struct is_dynamic_tensor<dynamic_tensor<T>> : std::true_type {};
//==============================================================================
// transpose
//==============================================================================
template <typename DynamicTensor> requires is_dynamic_tensor_v<DynamicTensor>
struct const_transposed_dynamic_tensor {
  using value_type = typename DynamicTensor::value_type;
  DynamicTensor  const& m_tensor;
  //============================================================================
  auto at(integral auto const... /*is*/) const -> value_type const& {
    throw std::runtime_error{
        "[const_transposed_dynamic_tensor::at] need exactly two indices"};
  }
  auto at(size_t const r, size_t const c) const -> value_type const& {
    return m_tensor(c, r);
  }
  //----------------------------------------------------------------------------
  auto operator()(integral auto const... is) const -> value_type const& {
    if (sizeof...(is) == 2) {
      return at(is...);
    }
    throw std::runtime_error{
        "[const_transposed_dynamic_tensor::operator()] need exactly two "
        "indices"};
  }
  //----------------------------------------------------------------------------
  static constexpr auto num_dimensions() { return 2; }
  //----------------------------------------------------------------------------
  auto size(size_t const i) const { return m_tensor.size(1 - i); }
};
//------------------------------------------------------------------------------
template <typename DynamicTensor> requires is_dynamic_tensor_v<DynamicTensor>
struct is_dynamic_tensor<const_transposed_dynamic_tensor<DynamicTensor>>
    : std::true_type {};
//==============================================================================
template <typename DynamicTensor>
struct transposed_dynamic_tensor {
  using value_type = typename DynamicTensor::value_type;
  DynamicTensor & m_tensor;
  //============================================================================
  auto at(integral auto const... /*is*/) const -> value_type const& {
    throw std::runtime_error{
        "[transposed_dynamic_tensor::at] need exactly two indices"};
  }
  auto at(integral auto const... /*is*/) -> value_type& {
    throw std::runtime_error{
        "[transposed_dynamic_tensor::at] need exactly two indices"};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto at(size_t const r, size_t const c) const -> value_type const& {
    return m_tensor(c, r);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto at(size_t const r, size_t const c) -> value_type& {
    return m_tensor(c, r);
  }
  //----------------------------------------------------------------------------
  auto operator()(integral auto const... is) const -> value_type const& {
    return at(is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator()(integral auto const... is) -> value_type& { return at(is...); }
  //----------------------------------------------------------------------------
  static constexpr auto num_dimensions() { return 2; }
  //----------------------------------------------------------------------------
  auto size(size_t const i) const { return m_tensor.size(1 - i); }
};
//------------------------------------------------------------------------------
template <typename DynamicTensor> requires is_dynamic_tensor_v<DynamicTensor>
struct is_dynamic_tensor<transposed_dynamic_tensor<DynamicTensor>>
    : std::true_type {};
//==============================================================================
template <typename DynamicTensor> requires is_dynamic_tensor_v<DynamicTensor>
auto transposed(DynamicTensor const& A) {
  assert(A.num_dimensions() == 2);
  return const_transposed_dynamic_tensor<DynamicTensor>{A};
}
//------------------------------------------------------------------------------
template <typename DynamicTensor> requires is_dynamic_tensor_v<DynamicTensor>
auto transposed(DynamicTensor& A) {
  assert(A.num_dimensions() == 2);
  return transposed_dynamic_tensor<DynamicTensor>{A};
}
//==============================================================================
// diag
//==============================================================================
template <typename DynamicTensor> requires is_dynamic_tensor_v<DynamicTensor>
struct const_diag_dynamic_tensor {
  using value_type = typename DynamicTensor::value_type;
  DynamicTensor const&  m_tensor;
  static constexpr auto zero = typename DynamicTensor::value_type{};
  //============================================================================
  auto at(integral auto const... /*is*/) const -> value_type const& {
    throw std::runtime_error{
        "[const_diag_dynamic_tensor::at] need exactly two indices"};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto at(size_t const r, size_t const c) const -> value_type const& {
    if (r == c) {
      return m_tensor(r);
    }
    return zero;
  }
  //----------------------------------------------------------------------------
  auto operator()(integral auto const ...is) const -> value_type const& {
    return at(is...);
  }
  //----------------------------------------------------------------------------
  static constexpr auto num_dimensions() { return 2; }
  //----------------------------------------------------------------------------
  auto size(size_t const /*i*/) const { return m_tensor.size(0); }
};
//------------------------------------------------------------------------------
template <typename DynamicTensor> requires is_dynamic_tensor_v<DynamicTensor>
struct is_dynamic_tensor<const_diag_dynamic_tensor<DynamicTensor>>
    : std::true_type {};
//==============================================================================
template <typename DynamicTensor>
struct diag_dynamic_tensor {
  using value_type = typename DynamicTensor::value_type;
   DynamicTensor& m_tensor;
  static constexpr auto zero = typename DynamicTensor::value_type{};
  //============================================================================
  auto at(integral auto const... /*is*/) const -> value_type const& {
    throw std::runtime_error{
        "[diag_dynamic_tensor::at] need exactly two indices"};
  }
  auto at(integral auto const... /*is*/) -> value_type& {
    throw std::runtime_error{
        "[diag_dynamic_tensor::at] need exactly two indices"};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto at(size_t const r, size_t const c) const -> value_type const& {
    if (r == c) {
      return m_tensor(r);
    }
    return zero;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto at(size_t const r, size_t const c) -> value_type& {
    static typename DynamicTensor::value_type zero;
    zero = typename DynamicTensor::value_type{};
    if (r == c) {
      return m_tensor(r);
    }
    return zero;
  }
  //----------------------------------------------------------------------------
  auto operator()(integral auto const... is) const -> value_type const& {
    return at(is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator()(integral auto const... is) -> value_type& { return at(is...); }
  //----------------------------------------------------------------------------
  static constexpr auto num_dimensions() { return 2; }
  //----------------------------------------------------------------------------
  auto size(size_t const /*i*/) const { return m_tensor.size(0); }
};
//------------------------------------------------------------------------------
template <typename DynamicTensor> requires is_dynamic_tensor_v<DynamicTensor>
struct is_dynamic_tensor<diag_dynamic_tensor<DynamicTensor>>
    : std::true_type {};
//==============================================================================
template <typename DynamicTensor> requires is_dynamic_tensor_v<DynamicTensor>
auto diag(DynamicTensor const& A) {
  assert(A.num_dimensions() == 1);
  return const_diag_dynamic_tensor<DynamicTensor>{A};
}
//------------------------------------------------------------------------------
template <typename DynamicTensor> requires is_dynamic_tensor_v<DynamicTensor>
auto diag(DynamicTensor& A) {
  assert(A.num_dimensions() == 1);
  return diag_dynamic_tensor<DynamicTensor>{A};
}
//==============================================================================
template <typename LhsTensor, typename RhsTensor>
requires is_dynamic_tensor_v<LhsTensor> &&
         is_dynamic_tensor_v<RhsTensor>
auto operator*(LhsTensor const& lhs, RhsTensor const& rhs)
    -> dynamic_tensor<std::common_type_t<typename LhsTensor::value_type,
                                         typename RhsTensor::value_type>> {
  using out_t =
      dynamic_tensor<std::common_type_t<typename LhsTensor::value_type,
                                        typename RhsTensor::value_type>>;
  out_t out;
  // matrix-matrix-multiplication
  if (lhs.num_dimensions() == 2 && rhs.num_dimensions() == 2 &&
      lhs.size(1) == rhs.size(0)) {
    auto out = out_t::zeros(lhs.size(0), rhs.size(1));
    for (size_t r = 0; r < lhs.size(0); ++r) {
      for (size_t c = 0; c < rhs.size(1); ++c) {
        for (size_t i = 0; i < lhs.size(1); ++i) {
          out(r, c) += lhs(r, i) * rhs(i, c);
        }
      }
    }
    return out;
  }
  // matrix-vector-multiplication
  else if (lhs.num_dimensions() == 2 && rhs.num_dimensions() == 1 &&
           lhs.size(1) == rhs.size(0)) {
    auto out = out_t::zeros(lhs.size(0));
    for (size_t r = 0; r < lhs.size(0); ++r) {
      for (size_t c = 0; c < rhs.size(1); ++c) {
        for (size_t i = 0; i < lhs.size(1); ++i) {
          out(r) += lhs(r, i) * rhs(c);
        }
      }
    }
    return out;
  }

  std::stringstream A;
  A << "[ " << lhs.size(0);
  for (size_t i = 1; i < lhs.num_dimensions(); ++i) {
    A << " x " << lhs.size(i);
  }
  A << " ]";
      std::stringstream B;
  B << "[ " << rhs.size(0);
  for (size_t i = 1; i < rhs.num_dimensions(); ++i) {
    B << " x " << rhs.size(i);
  }
  B << " ]";
  throw std::runtime_error{"Cannot contract given dynamic tensors. (A:" +
                           A.str() + "; B" + B.str() + ")"};
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#include <tatooine/dynamic_lapack.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <real_or_complex_number T>
auto solve(dynamic_tensor<T> const& A, dynamic_tensor<T> const& b) {
  return lapack::gesv(A, b);
}
//==============================================================================
/// printing vector
template <typename DynamicTensor>
requires is_dynamic_tensor_v<DynamicTensor> auto operator<<(
    std::ostream& out, DynamicTensor const& v) -> auto& {
  if (v.num_dimensions() == 1) {
  out << "[ ";
  out << std::scientific;
  for (size_t i = 0; i < v.size(0); ++i) {
    if constexpr (!is_complex_v<typename DynamicTensor::value_type>) {
    }
    out << v(i) << ' ';
  }
  out << "]";
  out << std::defaultfloat;
  }
  return out;
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
