#ifndef TATOOINE_DYNAMIC_TENSOR_H
#define TATOOINE_DYNAMIC_TENSOR_H
//==============================================================================
#include <tatooine/multidim_array.h>

#include <ostream>
#include <sstream>
//==============================================================================
namespace tatooine {
//==============================================================================
template <typename T>
struct is_dynamic_tensor_impl : std::false_type {};
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename T>
static constexpr auto is_dynamic_tensor = is_dynamic_tensor_impl<T>::value;
//==============================================================================
template <arithmetic_or_complex T>
struct tensor<T> : dynamic_multidim_array<T> {
  using this_t   = tensor<T>;
  using parent_t = dynamic_multidim_array<T>;
  using parent_t::parent_t;
  //============================================================================
  // factories
  //============================================================================
  static auto zeros(integral auto const... size) {
    return this_t{tag::zeros, size...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  static auto zeros(integral_range auto const& size) {
    return this_t{tag::zeros, size};
  }
  //----------------------------------------------------------------------------
  static auto ones(integral auto... size) {
    return this_t{tag::ones, size...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  static auto ones(integral_range auto const& size) {
    return this_t{tag::ones, size};
  }
  //------------------------------------------------------------------------------
  template <typename RandEng = std::mt19937_64>
  static auto randu(T const min, T const max, integral_range auto const& size,
                    RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_t{
        random::uniform<T, RandEng>{min, max, std::forward<RandEng>(eng)},
        size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename RandEng = std::mt19937_64>
  static auto randu(integral_range auto const& size, T min = 0, T max = 1,
                    RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_t{
        random::uniform<T, RandEng>{min, max, std::forward<RandEng>(eng)},
        size};
  }
  //----------------------------------------------------------------------------
  template <typename RandEng>
  static auto rand(random::uniform<T, RandEng> const& rand,
                   integral_range auto const&           size) {
    return this_t{rand, size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename RandEng>
  static auto rand(random::uniform<T, RandEng> const& rand, integral auto const... size) {
    return this_t{rand, std::vector{static_cast<size_t>(size)...}};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename RandEng>
  static auto rand(random::uniform<T, RandEng>&& rand,
                   integral_range auto const&    size) {
    return this_t{std::move(rand), size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <size_t N, integral Size, typename RandEng>
  static auto rand(random::uniform<T, RandEng>&& rand,
                   std::array<Size, N> const&    size) {
    return this_t{std::move(rand), size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename RandEng>
  static auto rand(random::uniform<T, RandEng>&& rand, integral auto const... size) {
    return this_t{std::move(rand), std::vector{static_cast<size_t>(size)...}};
  }
  //----------------------------------------------------------------------------
  template <typename RandEng>
  static auto rand(random::normal<T, RandEng> const& rand,
                   integral_range auto const&          size) {
    return this_t{rand, size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename RandEng>
  static auto rand(random::normal<T, RandEng> const& rand, integral auto const... size) {
    return this_t{rand, std::vector{static_cast<size_t>(size)...}};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename RandEng>
  static auto rand(random::normal<T, RandEng>&& rand,
                   integral auto const&     size) {
    return this_t{std::move(rand), size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename RandEng>
  static auto rand(random::normal<T, RandEng>&& rand, integral auto const... size) {
    return this_t{std::move(rand), std::vector{static_cast<size_t>(size)...}};
  }
  //----------------------------------------------------------------------------
  static auto vander(floating_point_range auto const& v) {
    return vander(v, v.size());
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  static auto vander(floating_point_range auto const& v, integral auto const degree) {
    auto V = this_t{v.size(), degree};
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
  // constructors
  //============================================================================
  constexpr tensor(integral auto const... dimensions) {
    this->resize(dimensions...);
  }
  //============================================================================
  // operators
  //============================================================================
  template <typename OtherTensor>
  requires is_dynamic_tensor<OtherTensor>
  auto operator=(OtherTensor const& other) -> tensor<T>& {
    if constexpr (is_transposed_tensor<OtherTensor>) {
      if (this == &other.internal_tensor()) {
        auto const old_size = dynamic_multidim_size{*this};
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

  template <typename OtherTensor>
  requires is_dynamic_tensor<OtherTensor>
  auto assign(OtherTensor const& other) {
    this->resize(other.size());
    auto const s = this->size();
    auto const r = this->num_dimensions();
    auto       max_cnt =
        std::accumulate(begin(s), end(s), size_t(1), std::multiplies<size_t>{});
    auto cnt = size_t(0);
    auto is  = std::vector<size_t>(r, 0);

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
  //============================================================================
  // methods
  //============================================================================
  auto dimension(size_t const i) const { return this->size(i); }
  auto rank() const { return this->num_dimensions(); }
};
template <typename... Rows, size_t N>
tensor(Rows(&&... rows)[N]) -> tensor<common_type<Rows...>>;

template <typename... Ts>
tensor(Ts...) -> tensor<common_type<Ts...>>;
//==============================================================================
template <typename T>
struct is_dynamic_tensor_impl<tensor<T>> : std::true_type {};
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
