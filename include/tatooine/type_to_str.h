#ifndef TATOOINE_TYPE_TO_STR_H
#define TATOOINE_TYPE_TO_STR_H

//==============================================================================
namespace tatooine {
//==============================================================================

template <typename>
struct type_to_str_false_type : std::false_type {};

#if has_cxx17_support()
template <typename Data>
constexpr std::string_view type_to_str() {
  static_assert(type_to_str_false_type<Data>::value, "unknown type");
  return "";
}
//-----------------------------------------------------------------------------
template <>
constexpr std::string_view type_to_str<double>() {
  return "double";
}
//-----------------------------------------------------------------------------
template <>
constexpr std::string_view type_to_str<long double>() {
  return "long double";
}
//-----------------------------------------------------------------------------
template <>
constexpr std::string_view type_to_str<float>() {
  return "float";
}
//-----------------------------------------------------------------------------
template <>
constexpr std::string_view type_to_str<int>() {
  return "int";
}
//-----------------------------------------------------------------------------
template <>
constexpr std::string_view type_to_str<unsigned int>() {
  return "unsigned int";
}
//-----------------------------------------------------------------------------
template <>
constexpr std::string_view type_to_str<long>() {
  return "long";
}
//-----------------------------------------------------------------------------
template <>
constexpr std::string_view type_to_str<unsigned long>() {
  return "unsigned long";
}
//-----------------------------------------------------------------------------
template <>
constexpr std::string_view type_to_str<long long>() {
  return "long long";
}
//-----------------------------------------------------------------------------
template <>
constexpr std::string_view type_to_str<unsigned long long>() {
  return "unsigned long long";
}
//-----------------------------------------------------------------------------
template <>
constexpr std::string_view type_to_str<char>() {
  return "char";
}
//-----------------------------------------------------------------------------
template <>
constexpr std::string_view type_to_str<unsigned char>() {
  return "unsigned char";
}
#else
template <typename Data>
std::string type_to_str() {
  static_assert(type_to_str_false_type<Data>::value, "unknown type");
  return "";
}
//-----------------------------------------------------------------------------
template <>
std::string type_to_str<double>() {
  return "double";
}
//-----------------------------------------------------------------------------
template <>
std::string type_to_str<long double>() {
  return "long double";
}
//-----------------------------------------------------------------------------
template <>
std::string type_to_str<float>() {
  return "float";
}
//-----------------------------------------------------------------------------
template <>
std::string type_to_str<int>() {
  return "int";
}
//-----------------------------------------------------------------------------
template <>
std::string type_to_str<unsigned int>() {
  return "unsigned int";
}
//-----------------------------------------------------------------------------
template <>
std::string type_to_str<long>() {
  return "long";
}
//-----------------------------------------------------------------------------
template <>
std::string type_to_str<unsigned long>() {
  return "unsigned long";
}
//-----------------------------------------------------------------------------
template <>
std::string type_to_str<long long>() {
  return "long long";
}
//-----------------------------------------------------------------------------
template <>
std::string type_to_str<unsigned long long>() {
  return "unsigned long long";
}
//-----------------------------------------------------------------------------
template <>
std::string type_to_str<char>() {
  return "char";
}
//-----------------------------------------------------------------------------
template <>
std::string type_to_str<unsigned char>() {
  return "unsigned char";
}

#endif

//==============================================================================
}  // namespace tatooine
//==============================================================================
#endif
