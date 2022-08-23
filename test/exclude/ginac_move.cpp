#include <ginac/ginac.h>
#include <catch2/catch_test_macros.hpp>

struct S {
  S() = default;

  S(const S&) = default;
  S& operator=(const S&) = default;

  //S(S&&)            = delete;
  //S&        operator=(S&&) = delete;
  GiNaC::ex ex;
};

struct T : S {
  T() = default;

  T(const T&) = default;
  T& operator=(const T&) = default;

  //T(T&&)     = delete;
  //T& operator=(T&&) = delete;
};

TEST_CASE("ginac_move") {
  T t;
  S s = std::move(t);
  s = std::move(t);
}
