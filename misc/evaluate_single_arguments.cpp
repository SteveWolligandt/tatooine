#include <array>
#include <concepts>
#include <iostream>

template <typename Head, typename... Tail>
decltype(auto) last(Head&& head, Tail&&... tail) {
  if constexpr (sizeof...(Tail) > 0) {
    return last(std::forward<Tail>(tail)...);
  } else {
    return std::forward<Head>(head);
  }
}

template <unsigned i, size_t N, typename Head, typename... Tail>
auto& assign(std::array<Head, N>& arr, Head const head, Tail const... tail) {
  if constexpr (i < N) {
    arr[i] = head;
    return assign<i + 1>(arr, tail...);
  } else {
    return arr;
  }
}

template <unsigned last_n, typename Head, typename... Tail>
auto remove_last_n(Head const head, Tail const... tail) {
  constexpr size_t       size = 1 + sizeof...(Tail) - last_n;
  std::array<Head, size> arr;
  return assign<0>(arr, head, tail...);
}

template <unsigned N>
struct V {
  auto operator()(std::array<int, N> const& x, int const t) const {
    std::cerr << x[0] << '\n';
    std::cerr << x[1] << '\n';
    std::cerr << t << '\n';
  }
  template <std::integral... XsT>
  auto operator()(XsT const... xst) const {
    static_assert(sizeof...(XsT) == N || sizeof...(XsT) == N + 1);
    if constexpr (sizeof...(XsT) == N) {
      (*this)(std::array{xst...}, 0);
    } else if constexpr (sizeof...(XsT) == N + 1) {
      (*this)(remove_last_n<1>(xst...), last(xst...));
    }
  }
};

int main() {
  V<2> v;
  v(1, 2, 3);
}
