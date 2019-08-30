#ifndef __TATOOINE_STRING_CONVERSION_H__
#define __TATOOINE_STRING_CONVERSION_H__

//==============================================================================
namespace tatooine {
//==============================================================================

template <typename real_t>
inline real_t parse(const std::string & /*to_parse*/);

//-----------------------------------------------------------------------------

template <>
inline float parse<float>(const std::string &to_parse) {
  return std::stof(to_parse);
}

//-----------------------------------------------------------------------------

template <>
inline double parse<double>(const std::string &to_parse) {
  return std::stod(to_parse);
}

//-----------------------------------------------------------------------------

template <>
inline long double parse<long double>(const std::string &to_parse) {
  return std::stold(to_parse);
}

//-----------------------------------------------------------------------------

template <>
inline int parse<int>(const std::string &to_parse) {
  return std::stoi(to_parse);
}

//-----------------------------------------------------------------------------

template <>
inline unsigned int parse<unsigned int>(const std::string &to_parse) {
  return (unsigned int)(std::stoul(to_parse));
}

//-----------------------------------------------------------------------------

template <>
inline long parse<long>(const std::string &to_parse) {
  return std::stol(to_parse);
}

//-----------------------------------------------------------------------------

template <>
inline unsigned long parse<unsigned long>(const std::string &to_parse) {
  return std::stoul(to_parse);
}

//-----------------------------------------------------------------------------

template <>
inline long long parse<long long>(const std::string &to_parse) {
  return std::stoll(to_parse);
}

//-----------------------------------------------------------------------------

template <>
inline unsigned long long parse<unsigned long long>(
    const std::string &to_parse) {
  return std::stoull(to_parse);
}

//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
