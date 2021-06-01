#ifndef TATOOINE_DYNAMIC_TENSOR_H
#define TATOOINE_DYNAMIC_TENSOR_H
//==============================================================================
#include <tatooine/is_transposed_tensor.h>
#include <tatooine/multidim_array.h>

#include <ostream>
#include <sstream>
//==============================================================================
namespace tatooine{
//==============================================================================
template <typename T>
struct is_dynamic_tensor_impl : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
static constexpr auto is_dynamic_tensor = is_dynamic_tensor_impl<T>::value;
//==============================================================================
#ifdef __cpp_concepts
template <arithmetic_or_complex T>
#else
template <typename T>
#endif
struct dynamic_tensor : dynamic_multidim_array<T> {
#ifndef __cpp_concepts
  static_assert(is_arithmetic<T> || is_complex<T>);
#endif
  using this_t   = dynamic_tensor<T>;
  using parent_t = dynamic_multidim_array<T>;
  using parent_t::parent_t;
  //============================================================================
  // factories
  //============================================================================
#ifdef __cpp_concepts
  template <integral... Size>
#else
  template <typename... Size, enable_if<is_integral<Size>...> = true>
#endif
  static auto zeros(Size const... size) {
    return this_t{tag::zeros, size...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <integral Size>
#else
  template <typename Size, enable_if<is_integral<Size>> = true>
#endif
  static auto zeros(std::vector<Size> const& size) {
    return this_t{tag::zeros, size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <integral Size, size_t N>
#else
  template <typename Size, size_t N, enable_if<is_integral<Size>> = true>
#endif
  static auto zeros(std::array<Size, N> const& size) {
    return this_t{tag::zeros, size};
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral... Size>
#else
  template <typename... Size, enable_if<is_integral<Size>...> = true>
#endif
  static auto ones(Size... size) { return this_t{tag::ones, size...}; }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <integral Size>
#else
  template <typename Size, enable_if<is_integral<Size>> = true>
#endif
  static auto ones(std::vector<Size> const& size) {
    return this_t{tag::ones, size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <integral Size, size_t N>
#else
  template <typename Size, size_t N, enable_if<is_integral<Size>> = true>
#endif
  static auto ones(std::array<Size, N> const& size) {
    return this_t{tag::ones, size};
  }
  //------------------------------------------------------------------------------
  // template <integral Size, typename RandEng = std::mt19937_64>
  // static auto randu(T min, T max, std::initializer_list<Size>&& size,
  //                  RandEng&& eng = RandEng{std::random_device{}()}) {
  //  return this_t{random_uniform{min, max, std::forward<RandEng>(eng)},
  //                std::vector<Size>(std::move(size))};
  //}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  // template <integral Size, typename RandEng = std::mt19937_64>
  // static auto randu(std::initializer_list<Size>&& size, T min = 0, T
  // max = 1,
  //                  RandEng&& eng = RandEng{std::random_device{}()}) {
  //  return this_t{random_uniform{min, max, std::forward<RandEng>(eng)},
  //                std::vector<Size>(std::move(size))};
  //}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <integral Size, typename RandEng = std::mt19937_64>
#else
  template <typename Size, typename RandEng = std::mt19937_64,
            enable_if<is_integral<Size>> = true>
#endif
  static auto randu(T const min, T const max, std::vector<Size> const& size,
                    RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_t{
        random_uniform<T, RandEng>{min, max, std::forward<RandEng>(eng)}, size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <integral Size, typename RandEng = std::mt19937_64>
#else
  template <typename Size, typename RandEng = std::mt19937_64,
            enable_if<is_integral<Size>> = true>
#endif
  static auto randu(std::vector<Size> const& size, T min = 0, T max = 1,
                    RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_t{
        random_uniform<T, RandEng>{min, max, std::forward<RandEng>(eng)}, size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <size_t N, integral Size,
            typename RandEng = std::mt19937_64>
#else
  template <size_t N, typename Size, typename RandEng = std::mt19937_64,
            enable_if<is_integral<Size>> = true>
#endif
  static auto randu(T min, T max, std::array<Size, N> const& size,
                    RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_t{
        random_uniform<T, RandEng>{min, max, std::forward<RandEng>(eng)}, size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <size_t N, integral Size,
            typename RandEng = std::mt19937_64>
#else
  template <size_t N, typename Size, typename RandEng = std::mt19937_64,
            enable_if<is_integral<Size>> = true>
#endif
  static auto randu(std::array<Size, N> const& size, T min = 0, T max = 1,
                    RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_t{
        random_uniform<T, RandEng>{min, max, std::forward<RandEng>(eng)}, size};
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral Size, typename RandEng>
#else
  template <typename Size, typename RandEng,
            enable_if<is_integral<Size>> = true>
#endif
  static auto rand(random_uniform<T, RandEng> const& rand,
                   std::vector<Size> const&          size) {
    return this_t{rand, size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#ifdef __cpp_concepts
  template <size_t N, integral Size, typename RandEng>
#else
  template <size_t N, typename Size, typename RandEng,
            enable_if<is_integral<Size>> = true>
#endif
  static auto rand(random_uniform<T, RandEng> const& rand,
                   std::array<Size, N> const&        size) {
    return this_t{rand, size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <typename RandEng, integral... Size>
#else
  template <typename RandEng, typename... Size,
            enable_if<is_integral<Size...>> = true>
#endif
  static auto rand(random_uniform<T, RandEng> const& rand,
                   Size... size) {
    return this_t{rand, std::vector{static_cast<size_t>(size)...}};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#ifdef __cpp_concepts
  template <integral Size, typename RandEng>
#else
  template <typename Size, typename RandEng,
            enable_if<is_integral<Size>> = true>
#endif
  static auto rand(random_uniform<T, RandEng>&& rand,
                   std::vector<Size> const&     size) {
    return this_t{std::move(rand), size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <size_t N, integral Size, typename RandEng>
#else
  template <size_t N, typename Size, typename RandEng,
            enable_if<is_integral<Size>> = true>
#endif
  static auto rand(random_uniform<T, RandEng>&& rand,
                   std::array<Size, N> const&   size) {
    return this_t{std::move(rand), size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <typename RandEng, integral... Size>
#else
  template <typename RandEng, typename... Size,
            enable_if<is_integral<Size...>> = true>
#endif
  static auto rand(random_uniform<T, RandEng>&& rand, Size... size) {
    return this_t{std::move(rand), std::vector{static_cast<size_t>(size)...}};
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral Size, typename RandEng>
#else
  template <typename Size, typename RandEng,
            enable_if<is_integral<Size>> = true>
#endif
  static auto rand(random_normal<T, RandEng> const& rand,
                   std::vector<Size> const&         size) {
    return this_t{rand, size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#ifdef __cpp_concepts
  template <size_t N, integral Size, typename RandEng>
#else
  template <size_t N, typename Size, typename RandEng,
            enable_if<is_integral<Size>> = true>
#endif
  static auto rand(random_normal<T, RandEng> const& rand,
                   std::array<Size, N> const&       size) {
    return this_t{rand, size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <typename RandEng, integral... Size>
#else
  template <typename RandEng, typename... Size,
            enable_if<is_integral<Size...>> = true>
#endif
  static auto rand(random_normal<T, RandEng> const& rand,
                   Size... size) {
    return this_t{rand, std::vector{static_cast<size_t>(size)...}};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <integral Size, typename RandEng>
#else
  template <typename Size, typename RandEng,
            enable_if<is_integral<Size>> = true>
#endif
  static auto rand(random_normal<T, RandEng>&& rand,
                   std::vector<Size> const&    size) {
    return this_t{std::move(rand), size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <size_t N, integral Size, typename RandEng>
#else
  template <size_t N, typename Size, typename RandEng,
            enable_if<is_integral<Size>> = true>
#endif
  static auto rand(random_normal<T, RandEng>&& rand,
                   std::array<Size, N> const&  size) {
    return this_t{std::move(rand), size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <typename RandEng, integral... Size>
#else
  template <typename RandEng, typename... Size,
            enable_if<is_integral<Size...>> = true>
#endif
  static auto rand(random_normal<T, RandEng>&& rand, Size... size) {
    return this_t{std::move(rand), std::vector{static_cast<size_t>(size)...}};
  }
  //----------------------------------------------------------------------------
  template <typename S, enable_if<is_arithmetic<S>> = true>
  static auto vander(std::vector<S> const& v) {
    return vander(v, v.size());
  }
  template <typename S, enable_if<is_arithmetic<S>> = true>
  static auto vander(std::vector<S> const& v, size_t const degree) {
    this_t V{v.size(), degree};
    auto   factor_up_row = [row = 0ul, &V, degree](auto const x) mutable {
      V(row, 0) = 1;
      for (std::size_t col = 1; col < degree; ++col) {
        V(row, col) = V(row, col - 1) * x;
      }
      ++row;
    };
    for (size_t i = 0; i < degree; ++i) {
      factor_up_row(static_cast<T>(v[i]));
    }
    return V;
  }
  //============================================================================
  template <typename... Rows, size_t N>
  constexpr dynamic_tensor(
      Rows(&&... rows)[N]) {
    static_assert(((is_arithmetic<Rows> || is_complex<Rows>)&&...));
    this->resize(sizeof...(Rows), N);

    // lambda inserting row into data block
    auto insert_row = [r = 0UL, this](auto const& row) mutable {
      for (size_t c = 0; c < N; ++c) {
        this->at(r, c) = static_cast<T>(row[c]);
      }
      ++r;
    };

    for_each(insert_row, rows...);
  }
  //----------------------------------------------------------------------------
  template <typename OtherTensor,
            enable_if<is_dynamic_tensor<OtherTensor>> = true>
  auto operator=(OtherTensor const& other) -> dynamic_tensor<T>& {
    if constexpr (is_transposed_tensor_v<OtherTensor>) {
      if (this == &other.internal_tensor()) {
        auto const old_size = dynamic_multidim_size {*this};
        this->resize(other.size());
        for (size_t col = 0; col < this->size(1); ++col) {
          for (size_t row = col + 1; row < this->size(0); ++row) {
            std::swap(this->at(row, col),
                      this->data(old_size.plain_index(col, row)));
          }
        }
        return *this;
      } else {
        assign(other);
      }
    }
    assign(other);
    return *this;
  }

  template <typename OtherTensor,
            enable_if<is_dynamic_tensor<OtherTensor>> = true>
  auto assign(OtherTensor const& other) {
    this->resize(other.size());
    auto const s   = this->size();
    auto const r   = this->num_dimensions();
    auto       max_cnt = std::accumulate(begin(s), end(s), size_t(1), std::multiplies<size_t>{});
    auto       cnt = size_t(0);
    auto       is  = std::vector<size_t>(r, 0);

    while (cnt < max_cnt) {
      this->at(is) = other(is);

      ++is.front();
      for (size_t i = 0; i < r - 1; ++i) {
        if (is[i] == s[i]) {
          is[i] = 0;
          ++is[i + 1];
        } else {
          break;
        }
      }
      ++cnt;
    }
  }
};
//------------------------------------------------------------------------------
template <typename T>
struct is_dynamic_tensor_impl<dynamic_tensor<T>> : std::true_type {};
//==============================================================================
// transpose
//==============================================================================
template <typename DynamicTensor>
#ifdef __cpp_concepts
requires is_dynamic_tensor<DynamicTensor>
#endif
struct const_transposed_dynamic_tensor {
  using value_type = typename DynamicTensor::value_type;
  DynamicTensor  const& m_tensor;
  //============================================================================
  auto internal_tensor() -> auto& { return m_tensor; }
  auto internal_tensor() const -> auto const& { return m_tensor; }
  //----------------------------------------------------------------------------
  auto size() const {
    auto s = m_tensor.size();
    std::reverse(begin(s), end(s));
    return s;
  }
  //============================================================================
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if<is_integral<Is...>> = true>
#endif
  auto at(Is const... /*is*/) const -> value_type const& {
    throw std::runtime_error{
        "[const_transposed_dynamic_tensor::at] need exactly two indices"};
  }
#ifdef __cpp_concepts
  template <integral R, integral C>
#else
  template <typename R, typename C, enable_if<is_integral<R, C>> = true>
#endif
  auto at(R const r, C const c) const -> value_type const& {
    return m_tensor(c, r);
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if<is_integral<Is...>> = true>
#endif
  auto operator()(Is const... is) const -> value_type const& {
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
template <typename DynamicTensor>
struct is_dynamic_tensor_impl<const_transposed_dynamic_tensor<DynamicTensor>>
    : std::true_type {};
//------------------------------------------------------------------------------
template <typename DynamicTensor>
struct is_transposed_tensor<const_transposed_dynamic_tensor<DynamicTensor>>
    : std::true_type {};
//==============================================================================
template <typename DynamicTensor>
#ifdef __cpp_concepts
requires is_dynamic_tensor<DynamicTensor>
#endif
struct transposed_dynamic_tensor {
  using value_type = typename DynamicTensor::value_type;
  DynamicTensor & m_tensor;
  //============================================================================
  auto internal_tensor() -> auto& { return m_tensor; }
  auto internal_tensor() const -> auto const& { return m_tensor; }
  //----------------------------------------------------------------------------
  auto size() const {
    auto s = m_tensor.size();
    std::reverse(begin(s), end(s));
    return s;
  }
  //============================================================================
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if<is_integral<Is...>> = true>
#endif
  auto at(Is const... /*is*/) const -> value_type const& {
    throw std::runtime_error{
        "[transposed_dynamic_tensor::at] need exactly two indices"};
  }
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if<is_integral<Is...>> = true>
#endif
  auto at(Is const... /*is*/) -> value_type& {
    throw std::runtime_error{
        "[transposed_dynamic_tensor::at] need exactly two indices"};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <integral R, integral C>
#else
  template <typename R, typename C, enable_if<is_integral<R, C>> = true>
#endif
  auto at(R const r, C const c) const -> value_type const& {
    return m_tensor(c, r);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <range Indices>
#else
  template <typename Indices, enable_if<is_range<Indices>> = true>
#endif
  auto operator()(Indices const& indices) const -> auto const& {
    return at(indices);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <range Indices>
#else
  template <typename Indices, enable_if<is_range<Indices>> = true>
#endif
  auto operator()(Indices const& indices) -> auto& {
    return at(indices);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <range Indices>
#else
  template <typename Indices, enable_if<is_range<Indices>> = true>
#endif
  auto at(Indices indices) -> auto& {
    assert(indices.size() == num_dimensions());
    std::reverse(begin(indices), end(indices));
    return m_tensor(indices);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <range Indices>
#else
  template <typename Indices, enable_if<is_range<Indices>> = true>
#endif
  auto at(Indices indices) const -> auto const& {
    assert(indices.size() == num_dimensions());
    std::reverse(begin(indices), end(indices));
    return m_tensor(indices);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <integral R, integral C>
#else
  template <typename R, typename C, enable_if<is_integral<R, C>> = true>
#endif
  auto at(R const r, C const c) -> value_type& {
    return m_tensor(c, r);
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if<is_integral<Is...>> = true>
#endif
  auto operator()(Is const... is) const -> value_type const& {
    return at(is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if<is_integral<Is...>> = true>
#endif
  auto operator()(Is const... is) -> value_type& { return at(is...); }
  //----------------------------------------------------------------------------
  static constexpr auto num_dimensions() { return 2; }
  //----------------------------------------------------------------------------
  auto size(size_t const i) const { return m_tensor.size(1 - i); }
};
//------------------------------------------------------------------------------
template <typename DynamicTensor>
struct is_dynamic_tensor_impl<transposed_dynamic_tensor<DynamicTensor>>
    : std::true_type {};
//------------------------------------------------------------------------------
template <typename DynamicTensor>
struct is_transposed_tensor<transposed_dynamic_tensor<DynamicTensor>>
    : std::true_type {};
//==============================================================================
#ifdef __cpp_concepts
template <typename DynamicTensor>
requires is_dynamic_tensor<DynamicTensor>
#else
template <typename DynamicTensor,
          enable_if<is_dynamic_tensor<DynamicTensor>> = true>
#endif
auto transposed(DynamicTensor const& A) {
  assert(A.num_dimensions() == 2);
  return const_transposed_dynamic_tensor<DynamicTensor>{A};
}
//------------------------------------------------------------------------------
#ifdef __cpp_concepts
template <typename DynamicTensor>
requires is_dynamic_tensor<DynamicTensor>
#else
template <typename DynamicTensor,
          enable_if<is_dynamic_tensor<DynamicTensor>> = true>
#endif
auto transposed(DynamicTensor& A) {
  assert(A.num_dimensions() == 2);
  return transposed_dynamic_tensor<DynamicTensor>{A};
}
//==============================================================================
// diag
//==============================================================================
template <typename DynamicTensor>
#ifdef __cpp_concepts
requires is_dynamic_tensor<DynamicTensor>
#endif
struct const_diag_dynamic_tensor {
  using value_type = typename DynamicTensor::value_type;
  DynamicTensor const&  m_tensor;
  static constexpr auto zero = typename DynamicTensor::value_type{};
  //============================================================================
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if<is_integral<Is...>> = true>
#endif
  auto at(Is const... /*is*/) const -> value_type const& {
    throw std::runtime_error{
        "[const_diag_dynamic_tensor::at] need exactly two indices"};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <integral R, integral C>
#else
  template <typename R, typename C, enable_if<is_integral<R, C>> = true>
#endif
  auto at(R const r, C const c) const -> value_type const& {
    if (r == c) {
      return m_tensor(r);
    }
    return zero;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <range Indices>
#else
  template <typename Indices, enable_if<is_range<Indices>> = true>
#endif
  auto operator()(Indices const& indices) const -> auto const& {
    return at(indices);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <range Indices>
#else
  template <typename Indices, enable_if<is_range<Indices>> = true>
#endif
  auto operator()(Indices const& indices) -> auto& {
    return at(indices);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <range Indices>
#else
  template <typename Indices, enable_if<is_range<Indices>> = true>
#endif
  auto at(Indices indices) -> auto& {
    assert(indices.size() == num_dimensions());
    std::reverse(begin(indices), end(indices));
    return m_tensor(indices);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <range Indices>
#else
  template <typename Indices, enable_if<is_range<Indices>> = true>
#endif
  auto at(Indices indices) const -> auto const& {
    assert(indices.size() == num_dimensions());
    std::reverse(begin(indices), end(indices));
    return m_tensor(indices);
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if<is_integral<Is...>> = true>
#endif
  auto operator()(Is const ...is) const -> value_type const& {
    return at(is...);
  }
  //----------------------------------------------------------------------------
  static constexpr auto num_dimensions() { return 2; }
  //----------------------------------------------------------------------------
  auto size(size_t const /*i*/) const { return m_tensor.size(0); }
};
//------------------------------------------------------------------------------
template <typename DynamicTensor>
#ifdef __cpp_concepts
requires is_dynamic_tensor<DynamicTensor>
#endif
struct is_dynamic_tensor_impl<const_diag_dynamic_tensor<DynamicTensor>>
    : std::true_type {};
//==============================================================================
template <typename DynamicTensor>
struct diag_dynamic_tensor {
  using value_type = typename DynamicTensor::value_type;
   DynamicTensor& m_tensor;
  static constexpr auto zero = typename DynamicTensor::value_type{};
  //============================================================================
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if<is_integral<Is...>> = true>
#endif
  auto at(Is const... /*is*/) const -> value_type const& {
    throw std::runtime_error{
        "[diag_dynamic_tensor::at] need exactly two indices"};
  }
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if<is_integral<Is...>> = true>
#endif
  auto at(Is const... /*is*/) -> value_type& {
    throw std::runtime_error{
        "[diag_dynamic_tensor::at] need exactly two indices"};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <integral R, integral C>
#else
  template <typename R, typename C, enable_if<is_integral<R, C>> = true>
#endif
  auto at(R const r, C const c) const -> value_type const& {
    if (r == c) {
      return m_tensor(r);
    }
    return zero;
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <integral R, integral C>
#else
  template <typename R, typename C, enable_if<is_integral<R, C>> = true>
#endif
  auto at(R const r, C const c) -> value_type& {
    static typename DynamicTensor::value_type zero;
    zero = typename DynamicTensor::value_type{};
    if (r == c) {
      return m_tensor(r);
    }
    return zero;
  }
  //----------------------------------------------------------------------------
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if<is_integral<Is...>> = true>
#endif
  auto operator()(Is const... is) const -> value_type const& {
    return at(is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#ifdef __cpp_concepts
  template <integral... Is>
#else
  template <typename... Is, enable_if<is_integral<Is...>> = true>
#endif
  auto operator()(Is const... is) -> value_type& { return at(is...); }
  //----------------------------------------------------------------------------
  static constexpr auto num_dimensions() { return 2; }
  //----------------------------------------------------------------------------
  auto size(size_t const /*i*/) const { return m_tensor.size(0); }
};
//------------------------------------------------------------------------------
template <typename DynamicTensor>
#ifdef __cpp_concepts
requires is_dynamic_tensor<DynamicTensor>
#endif
struct is_dynamic_tensor_impl<diag_dynamic_tensor<DynamicTensor>>
    : std::true_type {};
//==============================================================================
template <typename DynamicTensor>
#ifdef __cpp_concepts
requires is_dynamic_tensor<DynamicTensor>
#endif
auto diag(DynamicTensor const& A) {
  assert(A.num_dimensions() == 1);
  return const_diag_dynamic_tensor<DynamicTensor>{A};
}
//------------------------------------------------------------------------------
template <typename DynamicTensor>
#ifdef __cpp_concepts
requires is_dynamic_tensor<DynamicTensor>
#endif
auto diag(DynamicTensor& A) {
  assert(A.num_dimensions() == 1);
  return diag_dynamic_tensor<DynamicTensor>{A};
}
//==============================================================================
#ifdef __cpp_concepts
template <typename LhsTensor, typename RhsTensor>
requires is_dynamic_tensor<LhsTensor>
#else
template <typename LhsTensor, typename RhsTensor,
          enable_if<is_dynamic_tensor<LhsTensor>> = true>
#endif
auto operator*(LhsTensor const& lhs, diag_dynamic_tensor<RhsTensor> const& rhs)
    -> dynamic_tensor<std::common_type_t<typename LhsTensor::value_type,
                                         typename RhsTensor::value_type>> {
  using out_t =
      dynamic_tensor<std::common_type_t<typename LhsTensor::value_type,
                                        typename RhsTensor::value_type>>;
  out_t out;
  // matrix-matrix-multiplication
  if (lhs.num_dimensions() == 2 &&
      lhs.size(1) == rhs.size(0)) {
    auto out = out_t::zeros(lhs.size(0), rhs.size(1));
    for (size_t r = 0; r < lhs.size(0); ++r) {
      for (size_t c = 0; c < rhs.size(1); ++c) {
        out(r, c) = lhs(r, c) * rhs(c, c);
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
#ifdef __cpp_concepts
template <typename LhsTensor, typename RhsTensor>
requires is_dynamic_tensor<LhsTensor> &&
         is_dynamic_tensor<RhsTensor>
#else
template <typename LhsTensor, typename RhsTensor,
          enable_if<is_dynamic_tensor<LhsTensor>,
                    is_dynamic_tensor<RhsTensor>> = true>
#endif
auto operator*(LhsTensor const& lhs, RhsTensor const& rhs)
    -> dynamic_tensor<std::common_type_t<typename LhsTensor::value_type,
                                         typename RhsTensor::value_type>> {
  using out_t =
      dynamic_tensor<std::common_type_t<typename LhsTensor::value_type,
                                        typename RhsTensor::value_type>>;
  out_t out;
  // matrix-matrix-multiplication
  if (lhs.num_dimensions() == 2 &&
      rhs.num_dimensions() == 2 &&
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
  else if (lhs.num_dimensions() == 2 &&
           rhs.num_dimensions() == 1 &&
           lhs.size(1) == rhs.size(0)) {
    auto out = out_t::zeros(lhs.size(0));
    for (size_t r = 0; r < lhs.size(0); ++r) {
      for (size_t i = 0; i < lhs.size(1); ++i) {
        out(r) += lhs(r, i) * rhs(i);
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
#ifdef __cpp_concepts
template <arithmetic_or_complex T>
#else
template <typename T>
#endif
auto solve(dynamic_tensor<T> const& A, dynamic_tensor<T> const& b) {
  return lapack::gesv(A, b);
}
//==============================================================================
/// printing vector
#ifdef __cpp_concepts
template <typename DynamicTensor>
requires is_dynamic_tensor<DynamicTensor>
#else
template <typename DynamicTensor,
          enable_if<is_dynamic_tensor<DynamicTensor>> = true>
#endif
auto operator<<(
    std::ostream& out, DynamicTensor const& v) -> auto& {
  if (v.num_dimensions() == 1) {
  out << "[ ";
  out << std::scientific;
  for (size_t i = 0; i < v.size(0); ++i) {
    if constexpr (!is_complex<typename DynamicTensor::value_type>) {
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
