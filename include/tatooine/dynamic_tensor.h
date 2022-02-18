#ifndef TATOOINE_DYNAMIC_TENSOR_H
#define TATOOINE_DYNAMIC_TENSOR_H
//==============================================================================
#include <tatooine/multidim_array.h>
#include <tatooine/einstein_notation/indexed_dynamic_tensor.h>

#include <ostream>
#include <sstream>
//==============================================================================
namespace tatooine {
//==============================================================================
template <arithmetic_or_complex T>
struct tensor<T> : dynamic_multidim_array<T> {
  using this_type   = tensor<T>;
  using parent_type = dynamic_multidim_array<T>;
  template <einstein_notation::index... Is>
  using const_indexed_type = einstein_notation::indexed_dynamic_tensor<this_type const&, Is...>;
  template <einstein_notation::index... Is>
  using indexed_type = einstein_notation::indexed_dynamic_tensor<this_type&, Is...>;
  using parent_type::at;
  using parent_type::parent_type;
  using parent_type::operator();

  static auto constexpr is_tensor() { return true; }
  static auto constexpr is_dynamic() { return true; }
  //============================================================================
  // factories
  //============================================================================
  static auto zeros(integral auto const... size) {
    return this_type{tag::zeros, size...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  static auto zeros(integral_range auto const& size) {
    return this_type{tag::zeros, size};
  }
  //----------------------------------------------------------------------------
  static auto ones(integral auto... size) {
    return this_type{tag::ones, size...};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  static auto ones(integral_range auto const& size) {
    return this_type{tag::ones, size};
  }
  //------------------------------------------------------------------------------
  template <typename RandEng = std::mt19937_64>
  static auto randu(T const min, T const max, integral_range auto const& size,
                    RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_type{
        random::uniform<T, RandEng>{min, max, std::forward<RandEng>(eng)},
        size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename RandEng = std::mt19937_64>
  static auto randu(integral_range auto const& size, T min = 0, T max = 1,
                    RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_type{
        random::uniform<T, RandEng>{min, max, std::forward<RandEng>(eng)},
        size};
  }
  //----------------------------------------------------------------------------
  template <typename RandEng>
  static auto rand(random::uniform<T, RandEng> const& rand,
                   integral_range auto const&           size) {
    return this_type{rand, size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename RandEng>
  static auto rand(random::uniform<T, RandEng> const& rand, integral auto const... size) {
    return this_type{rand, std::vector{static_cast<std::size_t>(size)...}};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename RandEng>
  static auto rand(random::uniform<T, RandEng>&& rand,
                   integral_range auto const&    size) {
    return this_type{std::move(rand), size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <std::size_t N, integral Size, typename RandEng>
  static auto rand(random::uniform<T, RandEng>&& rand,
                   std::array<Size, N> const&    size) {
    return this_type{std::move(rand), size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename RandEng>
  static auto rand(random::uniform<T, RandEng>&& rand, integral auto const... size) {
    return this_type{std::move(rand), std::vector{static_cast<std::size_t>(size)...}};
  }
  //----------------------------------------------------------------------------
  template <typename RandEng>
  static auto rand(random::normal<T, RandEng> const& rand,
                   integral_range auto const&          size) {
    return this_type{rand, size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename RandEng>
  static auto rand(random::normal<T, RandEng> const& rand, integral auto const... size) {
    return this_type{rand, std::vector{static_cast<std::size_t>(size)...}};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename RandEng>
  static auto rand(random::normal<T, RandEng>&& rand,
                   integral auto const&     size) {
    return this_type{std::move(rand), size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename RandEng>
  static auto rand(random::normal<T, RandEng>&& rand, integral auto const... size) {
    return this_type{std::move(rand), std::vector{static_cast<std::size_t>(size)...}};
  }
  //----------------------------------------------------------------------------
  static auto vander(floating_point_range auto const& v) {
    return vander(v, v.size());
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  static auto vander(floating_point_range auto const& v, integral auto const degree) {
    auto V = this_type{v.size(), degree};
    auto   factor_up_row = [row = 0ul, &V, degree](auto const x) mutable {
      V(row, 0) = 1;
      for (std::size_t col = 1; col < degree; ++col) {
        V(row, col) = V(row, col - 1) * x;
      }
      ++row;
    };
    for (std::size_t i = 0; i < degree; ++i) {
      factor_up_row(static_cast<T>(v[i]));
    }
    return V;
  }
  //============================================================================
  // constructors
  //============================================================================
  explicit constexpr tensor(dynamic_tensor auto&& other) {
    assign(other);
  }
  //----------------------------------------------------------------------------
  explicit constexpr tensor(integral auto const... dimensions) {
    this->resize(dimensions...);
  }
  //============================================================================
  // operators
  //============================================================================
  template <general_tensor OtherTensor>
  auto operator=(OtherTensor const& other) -> tensor<T>& {
    if constexpr (transposed_tensor<OtherTensor>) {
      if (this == &other.internal_tensor()) {
        auto const old_size = dynamic_multidim_size{*this};
        this->resize(other.dimensions());
        for (std::size_t col = 0; col < this->size(1); ++col) {
          for (std::size_t row = col + 1; row < this->size(0); ++row) {
            std::swap(this->at(row, col),
                      this->data(old_size.plain_index(col, row)));
          }
        }
        return *this;
      } else {
        assign(other);
      }
    } else {
      assign(other);
    }
    return *this;
  }
  //----------------------------------------------------------------------------
  auto assign(general_tensor auto const& other) {
    this->resize(other.dimensions());
    auto const s = this->size();
    auto const r = this->num_dimensions();
    auto       max_cnt =
        std::accumulate(begin(s), end(s), std::size_t(1), std::multiplies<std::size_t>{});
    auto cnt = std::size_t(0);
    auto is  = std::vector<std::size_t>(r, 0);

    while (cnt < max_cnt) {
      this->at(is) = other(is);

      ++is.front();
      for (std::size_t i = 0; i < r - 1; ++i) {
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
  auto dimensions() const { return this->size(); }
  auto dimension(std::size_t const i) const { return dimensions()[i]; }
  auto rank() const { return this->num_dimensions(); }
  //----------------------------------------------------------------------------
  template <einstein_notation::index... Is>
  requires(sizeof...(Is) == rank()) auto constexpr at(Is const... /*is*/) {
    return indexed_type<Is...>{*this};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <einstein_notation::index... Is>
  requires(sizeof...(Is) ==
           rank()) auto constexpr at(Is const... /*is*/) const {
    return const_indexed_type<Is...>{*this};
  }
  //----------------------------------------------------------------------------
  template <einstein_notation::index... Is>
  requires(sizeof...(Is) == rank()) auto constexpr operator()(
      Is const... is) const {
    return at(is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <einstein_notation::index... Is>
  requires(sizeof...(Is) == rank()) auto constexpr operator()(Is const... is) {
    return at(is...);
  }
};
template <typename... Rows, std::size_t N>
tensor(Rows(&&... rows)[N]) -> tensor<common_type<Rows...>>;

template <typename... Ts>
tensor(Ts...) -> tensor<common_type<Ts...>>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
