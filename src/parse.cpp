#include <tatooine/parse.h>
//==============================================================================
namespace tatooine {
//==============================================================================
template <>
auto parse<float>(std::string const &to_parse) -> float {
  return std::stof(to_parse);
}
//-----------------------------------------------------------------------------
template <>
auto parse<double>(std::string const &to_parse) -> double {
  return std::stod(to_parse);
}
//-----------------------------------------------------------------------------
template <>
auto parse<long double>(std::string const &to_parse) -> long double {
  return std::stold(to_parse);
}
//-----------------------------------------------------------------------------
template <>
auto parse<int>(std::string const &to_parse) -> int {
  return std::stoi(to_parse);
}
//-----------------------------------------------------------------------------
template <>
auto parse<unsigned int>(std::string const &to_parse) -> unsigned int {
  return (unsigned int)(std::stoul(to_parse));
}
//-----------------------------------------------------------------------------
template <>
auto parse<long>(std::string const &to_parse) -> long {
  return std::stol(to_parse);
}
//-----------------------------------------------------------------------------
template <>
auto parse<unsigned long>(std::string const &to_parse) -> unsigned long {
  return std::stoul(to_parse);
}
//-----------------------------------------------------------------------------
template <>
auto parse<long long>(std::string const &to_parse) -> long long {
  return std::stoll(to_parse);
}
//-----------------------------------------------------------------------------
template <>
auto parse<unsigned long long>(std::string const &to_parse)
    -> unsigned long long {
  return std::stoull(to_parse);
}
//==============================================================================
}  // namespace tatooine
//==============================================================================
