#include <iostream>
//==============================================================================
namespace tatooine::mpi::fortran {
//==============================================================================
extern "C" auto hello_world() -> void { std::cerr << "hello world\n"; }
extern "C" auto add(int a, int b) -> int { return a + b; }
extern "C" auto add_matrix(int const* a, int const* b, int const s1,
                           int const s2) -> void {
  size_t idx = 0;
  for (size_t j = 0; j < s2; ++j) {
    for (size_t i = 0; i < s1; ++i) {
      std::cerr << "[" << idx << "]:" << a[idx] + b[idx] << ' ';
      idx++;
    }
    std::cerr << '\n';
  }
}
//==============================================================================
}  // namespace tatooine::mpi::fortran
//==============================================================================
