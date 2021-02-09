#include <iostream>
//==============================================================================
namespace tatooine::mpi::fortan {
extern "C" {
//==============================================================================
auto hello_world() -> void { std::cerr << "hello world\n"; }
//------------------------------------------------------------------------------
auto add(int a, int b) -> int { return a + b; }
//------------------------------------------------------------------------------
auto add_matrix(int const* a, int const* b, int const s1, int const s2)
    -> void {
  size_t idx = 0;
  for (size_t j = 0; j < s2; ++j) {
    for (size_t i = 0; i < s1; ++i) {
      std::cerr << "[" << idx << "]:" << a[idx] + b[idx] << ' ';
      idx++;
    }
    std::cerr << '\n';
  }
}
//------------------------------------------------------------------------------
auto multi_arr(int const* arr) -> void {
  std::cerr << "\n====================\n";
  std::cerr << "c++ multi_arr\n";
  std::cerr << arr[0];
  for (size_t i = 1; i < 2 * 2 * 2 * 2; ++i) {
    std::cerr << ", " << arr[i];
  }
  std::cerr << '\n';
}
//------------------------------------------------------------------------------
auto print_str(const char* str) -> void {
  std::cerr << str << '\n';
}
//==============================================================================
} // extern "C"
}  // namespace tatooine::mpi::fortan
//==============================================================================
