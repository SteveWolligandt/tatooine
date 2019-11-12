#ifndef RANDOMSEED_H
#define RANDOMSEED_H

//==============================================================================
#include <random>
#include <boost/range/algorithm/generate.hpp>
#include <string>

//==============================================================================
static constexpr std::string_view alphanum = 
  "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

//------------------------------------------------------------------------------
template <typename random_engine_t = std::mt19937_64>
inline auto random_char(
    random_engine_t&& random_engine = random_engine_t{std::random_device{}()}) {
  std::uniform_int_distribution<size_t> udist{0, alphanum.size() - 1};
  return alphanum[udist(random_engine)];
}

//------------------------------------------------------------------------------
template <typename random_engine_t = std::mt19937_64>
inline auto random_string(
    const int len,
    random_engine_t&& random_engine = random_engine_t{std::random_device{}()}) {
  std::string random_string(len, ' ');
  boost::generate(random_string, [&]{return random_char(random_engine);});
  return random_string;
}

#endif
