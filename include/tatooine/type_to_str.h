#ifndef __TATOOINE_TYPE_TO_STR_H__
#define __TATOOINE_TYPE_TO_STR_H__

//==============================================================================
namespace tatooine {
//==============================================================================

template <typename>
struct type_to_str_false_type : std::false_type {};

template <typename data_t>
inline std::string type_to_str() {
  static_assert(type_to_str_false_type<data_t>::value, "unknown type");
  return "";
}
//-----------------------------------------------------------------------------
template <>
inline std::string type_to_str<double>() {
  return "double";
}
//-----------------------------------------------------------------------------
template <>
inline std::string type_to_str<long double>() {
  return "long double";
}
//-----------------------------------------------------------------------------
template <>
inline std::string type_to_str<float>() {
  return "float";
}
//-----------------------------------------------------------------------------
template <>
inline std::string type_to_str<int>() {
  return "int";
}
//-----------------------------------------------------------------------------
template <>
inline std::string type_to_str<unsigned int>() {
  return "unsigned int";
}
//-----------------------------------------------------------------------------
template <>
inline std::string type_to_str<long>() {
  return "long";
}
//-----------------------------------------------------------------------------
template <>
inline std::string type_to_str<unsigned long>() {
  return "unsigned long";
}
//-----------------------------------------------------------------------------
template <>
inline std::string type_to_str<long long>() {
  return "long long";
}
//-----------------------------------------------------------------------------
template <>
inline std::string type_to_str<unsigned long long>() {
  return "unsigned long long";
}
//-----------------------------------------------------------------------------
template <>
inline std::string type_to_str<char>() {
  return "char";
}
//-----------------------------------------------------------------------------
template <>
inline std::string type_to_str<unsigned char>() {
  return "unsigned char";
}

//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
