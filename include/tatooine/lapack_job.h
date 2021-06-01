#ifndef TATOOINE_LAPACK_JOB_H
#define TATOOINE_LAPACK_JOB_H
//==============================================================================
namespace tatooine::lapack::job {
//==============================================================================
struct A_t {static constexpr char value = 'A'; }; static constexpr A_t A;
struct L_t {static constexpr char value = 'L'; }; static constexpr L_t L;
struct N_t {static constexpr char value = 'N'; }; static constexpr N_t N;
struct O_t {static constexpr char value = 'O'; }; static constexpr O_t O;
struct S_t {static constexpr char value = 'S'; }; static constexpr S_t S;
struct U_t {static constexpr char value = 'U'; }; static constexpr U_t U;
struct V_t {static constexpr char value = 'V'; }; static constexpr V_t V;

template <typename J>
constexpr auto to_char() {
  return J::value;
}
//==============================================================================
}  // namespace tatooine::lapack::job
//==============================================================================
#endif
