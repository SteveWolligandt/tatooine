#ifndef TATOOINE_FLOWEXPLORER_NODES_TEST_NODE_H
#define TATOOINE_FLOWEXPLORER_NODES_TEST_NODE_H
//==============================================================================
#include <tatooine/flowexplorer/ui/node.h>
#include <tatooine/reflection.h>
#include <tatooine/utility.h>
//==============================================================================
namespace tatooine::flowexplorer::nodes {
//==============================================================================
struct test_node : ui::node<test_node> {
 private:
  //============================================================================
  std::string           m_string_var;
  int                   m_int_var;
  float                 m_float_var;
  double                m_double_var;
  std::array<int, 2>    m_arr_int2_var;
  std::array<int, 3>    m_arr_int3_var;
  std::array<int, 4>    m_arr_int4_var;
  std::array<float, 2>  m_arr_float2_var;
  std::array<float, 3>  m_arr_float3_var;
  std::array<float, 4>  m_arr_float4_var;
  std::array<double, 2> m_arr_double2_var;
  std::array<double, 3> m_arr_double3_var;
  std::array<double, 4> m_arr_double4_var;
  vec<int, 2>           m_vec_int2_var;
  vec<int, 3>           m_vec_int3_var;
  vec<int, 4>           m_vec_int4_var;
  vec<float, 2>         m_vec_float2_var;
  vec<float, 3>         m_vec_float3_var;
  vec<float, 4>         m_vec_float4_var;
  vec<double, 2>        m_vec_double2_var;
  vec<double, 3>        m_vec_double3_var;
  vec<double, 4>        m_vec_double4_var;

 public:
  //============================================================================
  // CONSTRUCTORS
  //============================================================================
  test_node(flowexplorer::scene& s) : node<test_node>{"Test", s} {
    insert_input_pin<test_node>("input");
    insert_output_pin<test_node>("output");
  }

 public:
  //============================================================================
  // SETTER / GETTER
  //============================================================================
  auto string() -> auto& { return m_string_var; }
  auto string() const -> auto const& { return m_string_var; }
  auto int1() -> auto& { return m_int_var; }
  auto int1() const-> auto const& { return m_int_var; }
  auto int2_arr() -> auto& { return m_arr_int2_var; }
  auto int2_arr() const-> auto const& { return m_arr_int2_var; }
  auto int3_arr() -> auto& { return m_arr_int3_var; }
  auto int3_arr() const-> auto const& { return m_arr_int3_var; }
  auto int4_arr() -> auto& { return m_arr_int4_var; }
  auto int4_arr() const-> auto const& { return m_arr_int4_var; }
  auto float1() -> auto& { return m_float_var; }
  auto float1() const-> auto const& { return m_float_var; }
  auto float2_arr() -> auto& { return m_arr_float2_var; }
  auto float2_arr() const-> auto const& { return m_arr_float2_var; }
  auto float3_arr() -> auto& { return m_arr_float3_var; }
  auto float3_arr() const-> auto const& { return m_arr_float3_var; }
  auto float4_arr() -> auto& { return m_arr_float4_var; }
  auto float4_arr() const-> auto const& { return m_arr_float4_var; }
  auto double1() -> auto& { return m_double_var; }
  auto double1() const-> auto const& { return m_double_var; }
  auto double2_arr() -> auto& { return m_arr_double2_var; }
  auto double2_arr() const-> auto const& { return m_arr_double2_var; }
  auto double3_arr() -> auto& { return m_arr_double3_var; }
  auto double3_arr() const-> auto const& { return m_arr_double3_var; }
  auto double4_arr() -> auto& { return m_arr_double4_var; }
  auto double4_arr() const-> auto const& { return m_arr_double4_var; }
  auto int2_vec() -> auto& { return m_vec_int2_var; }
  auto int2_vec() const-> auto const& { return m_vec_int2_var; }
  auto int3_vec() -> auto& { return m_vec_int3_var; }
  auto int3_vec() const-> auto const& { return m_vec_int3_var; }
  auto int4_vec() -> auto& { return m_vec_int4_var; }
  auto int4_vec() const-> auto const& { return m_vec_int4_var; }
  auto float2_vec() -> auto& { return m_vec_float2_var; }
  auto float2_vec() const-> auto const& { return m_vec_float2_var; }
  auto float3_vec() -> auto& { return m_vec_float3_var; }
  auto float3_vec() const-> auto const& { return m_vec_float3_var; }
  auto float4_vec() -> auto& { return m_vec_float4_var; }
  auto float4_vec() const-> auto const& { return m_vec_float4_var; }
  auto double2_vec() -> auto& { return m_vec_double2_var; }
  auto double2_vec() const-> auto const& { return m_vec_double2_var; }
  auto double3_vec() -> auto& { return m_vec_double3_var; }
  auto double3_vec() const-> auto const& { return m_vec_double3_var; }
  auto double4_vec() -> auto& { return m_vec_double4_var; }
  auto double4_vec() const-> auto const& { return m_vec_double4_var; }
};
//==============================================================================
}  // namespace tatooine::flowexplorer::nodes
//==============================================================================
REGISTER_NODE(tatooine::flowexplorer::nodes::test_node,
              TATOOINE_REFLECTION_INSERT_GETTER(string),
              TATOOINE_REFLECTION_INSERT_GETTER(int1),
              TATOOINE_REFLECTION_INSERT_GETTER(int2_vec),
              TATOOINE_REFLECTION_INSERT_GETTER(int3_vec),
              TATOOINE_REFLECTION_INSERT_GETTER(int4_vec),
              TATOOINE_REFLECTION_INSERT_GETTER(int2_arr),
              TATOOINE_REFLECTION_INSERT_GETTER(int3_arr),
              TATOOINE_REFLECTION_INSERT_GETTER(int4_arr),
              TATOOINE_REFLECTION_INSERT_GETTER(float1),
              TATOOINE_REFLECTION_INSERT_GETTER(float2_vec),
              TATOOINE_REFLECTION_INSERT_GETTER(float3_vec),
              TATOOINE_REFLECTION_INSERT_GETTER(float4_vec),
              TATOOINE_REFLECTION_INSERT_GETTER(float2_arr),
              TATOOINE_REFLECTION_INSERT_GETTER(float3_arr),
              TATOOINE_REFLECTION_INSERT_GETTER(float4_arr),
              TATOOINE_REFLECTION_INSERT_GETTER(double1),
              TATOOINE_REFLECTION_INSERT_GETTER(double2_vec),
              TATOOINE_REFLECTION_INSERT_GETTER(double3_vec),
              TATOOINE_REFLECTION_INSERT_GETTER(double4_vec),
              TATOOINE_REFLECTION_INSERT_GETTER(double2_arr),
              TATOOINE_REFLECTION_INSERT_GETTER(double3_arr),
              TATOOINE_REFLECTION_INSERT_GETTER(double4_arr));
#endif
