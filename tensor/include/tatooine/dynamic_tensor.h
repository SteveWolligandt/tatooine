#ifndef TATOOINE_DYNAMIC_TENSOR_H
#define TATOOINE_DYNAMIC_TENSOR_H
//==============================================================================
#include <tatooine/einstein_notation/indexed_dynamic_tensor.h>
#include <tatooine/multidim_array.h>

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
  using const_indexed_type =
      einstein_notation::indexed_dynamic_tensor<this_type const&, Is...>;
  template <einstein_notation::index... Is>
  using indexed_type =
      einstein_notation::indexed_dynamic_tensor<this_type&, Is...>;
  using parent_type::at;
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
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename RandEng = std::mt19937_64>
  static auto randu(integral auto const... is) {
    return this_type{random::uniform<T, RandEng>{
                         T(0), T(1), RandEng{std::random_device{}()}},
                     is...};
  }
  //------------------------------------------------------------------------------
  template <typename RandEng = std::mt19937_64>
  static auto randn(T const mean, T const stddev,
                    integral_range auto const& size,
                    RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_type{
        random::uniform<T, RandEng>{mean, stddev, std::forward<RandEng>(eng)},
        size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename RandEng = std::mt19937_64>
  static auto randn(integral_range auto const& size, T mean = 0, T stddev = 1,
                    RandEng&& eng = RandEng{std::random_device{}()}) {
    return this_type{
        random::uniform<T, RandEng>{mean, stddev, std::forward<RandEng>(eng)},
        size};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <typename RandEng = std::mt19937_64>
  static auto randn(integral auto const... is) {
    return this_type{random::uniform<T, RandEng>{
                         T(1), T(1), RandEng{std::random_device{}()}},
                     is...};
  }
  //----------------------------------------------------------------------------
  template <random_number_generator Rand, integral_range Size>
  static auto rand(Rand&& rand, Size&& size) {
    return this_type{std::forward<Rand>(rand), std::forward<Size>(size)};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <random_number_generator Rand>
  static auto rand(Rand&& rand, integral auto const... size){
    return this_type{std::forward<Rand>(rand), size...};
  }
  //----------------------------------------------------------------------------
  static auto vander(arithmetic_range auto const& v) {
    return vander(v, v.size());
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  static auto vander(arithmetic_range auto const& v,
                     integral auto const          degree) {
    auto V             = this_type{v.size(), degree};
    auto factor_up_row = [row = 0ul, &V, degree](auto const x) mutable {
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
  tensor()                          = default;
  tensor(tensor const& other)       = default;
  tensor(tensor&& other) noexcept = default;
  auto operator=(tensor const& other) -> tensor& = default;
  auto operator=(tensor&& other) noexcept -> tensor& = default;
  ~tensor() = default;
  //============================================================================
  explicit tensor(dynamic_tensor auto&& other) { assign(other); }
  //----------------------------------------------------------------------------
  /// Resizes Tensor to given dimensions and initializes values to 0.
  /// Only works if value_type is non-integral.
  explicit tensor(integral auto const... dimensions)
  requires(!integral<T>) {
    this->resize(dimensions...);
  }
  //----------------------------------------------------------------------------
  tensor(tag::ones_t tag, integral auto... dimensions)
      : parent_type{tag, dimensions...} {}
  //----------------------------------------------------------------------------
  tensor(tag::ones_t tag, integral_range auto&& dimensions)
      : parent_type{tag, std::forward<decltype(dimensions)>(dimensions)} {}
  //----------------------------------------------------------------------------
  tensor(tag::zeros_t tag, integral auto... dimensions)
      : parent_type{tag, dimensions...} {}
  //----------------------------------------------------------------------------
  tensor(tag::zeros_t tag, integral_range auto&& dimensions)
      : parent_type{tag, std::forward<decltype(dimensions)>(dimensions)} {}
  //----------------------------------------------------------------------------
  template <random_number_generator Rand>
  tensor(Rand&& rand, integral auto const... dimensions)
      : parent_type{std::forward<Rand>(rand), dimensions...} {}
  //----------------------------------------------------------------------------
  template <random_number_generator Rand, integral_range Dimensions>
  tensor(Rand&& rand, Dimensions&& dimensions)
      : parent_type{std::forward<Rand>(rand),
                    std::forward<Dimensions>(dimensions)} {}
  //----------------------------------------------------------------------------
  /// Constructs a rank 1 tensor aka vector.
  template <convertible_to<T>... Components>
  explicit tensor(Components&&... components) requires(
      (!integral<std::decay_t<Components>> && ...))
      : parent_type{std::vector<T>{std::forward<Components>(components)...},
                    sizeof...(Components)} {}
  //----------------------------------------------------------------------------
  /// Constructs a rank 2 tensor aka matrix.
  template <arithmetic_or_complex... Rows, std::size_t N>
  explicit tensor(Rows(&&... rows)[N])
      : parent_type{sizeof...(Rows), N} {
    // lambda inserting row into data block
    auto insert_row = [r = std::size_t(0), this](auto const& row) mutable {
      for (std::size_t c = 0; c < N; ++c) {
        at(r, c) = static_cast<T>(row[c]);
      }
      ++r;
    };

    for_each(insert_row, rows...);
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
  auto at(Is const... /*is*/) {
    assert(rank() == 0 || sizeof...(Is) == rank());
    return indexed_type<Is...>{*this};
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  template <einstein_notation::index... Is>
  auto at(Is const... /*is*/) const {
    assert(rank() == 0 || sizeof...(Is) == rank());
    return const_indexed_type<Is...>{*this};
  }
  //----------------------------------------------------------------------------
  auto operator()(einstein_notation::index auto const... is) const 
  {
    assert(rank() == 0 || sizeof...(is) == rank());
    return at(is...);
  }
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  auto operator()(einstein_notation::index auto const... is) {
    assert(rank() == 0 || sizeof...(is) == rank());
    return at(is...);
  }
};
//==============================================================================
template <typename... Rows, std::size_t N>
tensor(Rows(&&... rows)[N]) -> tensor<common_type<Rows...>>;

template <typename... Ts>
tensor(Ts...) -> tensor<common_type<Ts...>>;

template <dynamic_tensor Tensor>
tensor(Tensor &&) -> tensor<tensor_value_type<Tensor>>;
//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
