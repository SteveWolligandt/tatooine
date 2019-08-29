#ifndef TATOOINE_RANDOM_H
#define TATOOINE_RANDOM_H

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
