#ifndef TATOOINE_RANDOM_H
#define TATOOINE_RANDOM_H

#include <random>

#include "type_traits.h"

//==============================================================================
template <typename Engine, typename Real>
struct random_uniform {
  using engine_t = Engine;
  using real_t   = Real;
  using distribution_t =
      std::conditional_t<std::is_floating_point_v<Real>,
                         std::uniform_real_distribution<Real>,
                         std::uniform_int_distribution<Real>>;

  //============================================================================
  random_uniform() : engine{std::random_device{}()}, distribution{0, 1} {}
  random_uniform(const random_uniform&)     = default;
  random_uniform(random_uniform&&) noexcept = default;
  random_uniform& operator=(const random_uniform&) = default;
  random_uniform& operator=(random_uniform&&) noexcept = default;

  //----------------------------------------------------------------------------
  random_uniform(const Engine& _engine) : engine{_engine}, distribution{0, 1} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  random_uniform(Engine&& _engine) : engine{std::move(_engine)}, distribution{0, 1} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  random_uniform(Real min, Real max)
      : engine{std::random_device{}()}, distribution{min, max} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  random_uniform(const Engine& _engine, Real min, Real max)
      : engine{_engine}, distribution{min, max} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  random_uniform(Engine&& _engine, Real min, Real max)
      : engine{std::move(_engine)}, distribution{min, max} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  random_uniform(Real min, Real max, const Engine& _engine)
      : engine{_engine}, distribution{min, max} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  random_uniform(Real min, Real max, Engine&& _engine)
      : engine{std::move(_engine)}, distribution{min, max} {}

  //============================================================================
 private:
  Engine         engine;
  distribution_t distribution;

  //============================================================================
 public:
  auto get() { return distribution(engine); }
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
random_uniform()->random_uniform<std::mt19937_64, double>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Engine>
random_uniform(Engine &&)->random_uniform<Engine, double>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real>
random_uniform(Real min, Real max)->random_uniform<std::mt19937_64, Real>;

//==============================================================================
template <typename Engine, typename Real>
struct random_normal {
  using engine_t = Engine;
  using real_t = Real;
  using distribution_t = std::normal_distribution<Real>;

  //============================================================================
  random_normal() : engine{std::random_device{}()}, distribution{0, 1} {}
  random_normal(const random_normal&)     = default;
  random_normal(random_normal&&) noexcept = default;
  random_normal& operator=(const random_normal&) = default;
  random_normal& operator=(random_normal&&) noexcept = default;

  //----------------------------------------------------------------------------
  random_normal(const Engine& _engine) : engine{_engine}, distribution{0, 1} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  random_normal(Engine&& _engine) : engine{std::move(_engine)}, distribution{0, 1} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  random_normal(Real mean, Real stddev)
      : engine{std::random_device{}()}, distribution{mean, stddev} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  random_normal(const Engine& _engine, Real mean, Real stddev)
      : engine{_engine}, distribution{mean, stddev} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  random_normal(Engine&& _engine, Real mean, Real stddev)
      : engine{std::move(_engine)}, distribution{mean, stddev} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  random_normal(Real mean, Real stddev, const Engine& _engine)
      : engine{_engine}, distribution{mean, stddev} {}
  // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  random_normal(Real mean, Real stddev, Engine&& _engine)
      : engine{std::move(_engine)}, distribution{mean, stddev} {}

  //============================================================================
 private:
  Engine         engine;
  distribution_t distribution;

  //============================================================================
 public:
  auto get() { return distribution(engine); }
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
random_normal()->random_normal<std::mt19937_64, double>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Engine>
random_normal(Engine &&)->random_normal<Engine, double>;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
template <typename Real>
random_normal(Real mean, Real stddev)->random_normal<std::mt19937_64, Real>;

//==============================================================================
template <typename Iterator, typename RandomEngine>
auto random_elem(Iterator begin, Iterator end, RandomEngine& eng) {
  if (begin == end) { return end; }
  auto size = static_cast<size_t>(distance(begin, end) - 1);
  std::uniform_int_distribution<size_t> rand{0, size};
  return next(begin, rand(eng));
}

//------------------------------------------------------------------------------
template <typename Range, typename RandomEngine>
auto random_elem(Range&& range, RandomEngine& eng) {
  return random_elem(begin(range), end(range), eng);
}

//==============================================================================
enum coin { HEADS, TAILS };
template <typename RandomEngine>
auto flip_coin(RandomEngine&& eng) {
  std::uniform_int_distribution coin{0,1};
  return coin(eng) == 0 ? HEADS : TAILS;
}

#endif
