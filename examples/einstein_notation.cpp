#include <tatooine/for_loop.h>
#include <tatooine/einstein_notation.h>

#include <iostream>
#include <tuple>
//==============================================================================
using namespace tatooine;
//==============================================================================
template <typename Tensor, size_t I>
struct tensor_size;

template <size_t... Dimensions>
struct tensor {
  using this_type = tensor;
  static auto constexpr rank() { return sizeof...(Dimensions); }
  template <size_t I>
  static auto size() {
    return tensor_size<tensor, I>::value;
  }
  template <typename... Indices>
  auto operator()(Indices... /*unused*/) {
    static_assert(sizeof...(Indices) == rank(),
                  "Number of indices differs from tensor rank.");
    return index::tensor<this_type, Indices...>{};
  }
};

template <size_t I, size_t... Dimensions>
struct tensor_size<tensor<Dimensions...>, I> : ith_num<I, Dimensions...> {};

auto main() -> int {
  using namespace einstein_notation;
  [[maybe_unused]] auto T3   = ::tensor<2>{};
  [[maybe_unused]] auto T33  = ::tensor<2, 2>{};
  [[maybe_unused]] auto T333 = ::tensor<2, 2, 2>{};
  // T33(i, j) = T33(i, k) * T33(j, k);
  // std::cerr << "A(i, l) = B(i, j, k) * C(l, j, k)\n";
  // T33(i, l)  = T333(i, j, k) * T333(l, j, k);
  T333(j, k, l) = T3(i) * T33(i, j) * T333(i, k, l);
  // T3(i) = T333(i, j, k) * T33(j, k);
}
