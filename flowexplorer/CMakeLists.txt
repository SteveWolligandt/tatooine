set(TATFE_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(TATFE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(TATFE_BIN_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
set(TATFE_SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
add_executable(
  flowexplorer
  src/spacetime_splitted_vectorfield.cpp
  src/field_from_file.cpp
  src/axis_aligned_bounding_box.cpp
  src/flowexplorer.cpp
  src/doublegyre.cpp
  src/abcflow.cpp
  src/saddle.cpp
  src/position.cpp
  src/duffing_oscillator.cpp
  src/single_pathline.cpp
  src/autonomous_particles_flowmap.cpp
  src/autonomous_particles_flowmap_evaluator.cpp
  src/random_pathlines.cpp
  src/rayleigh_benard_convection.cpp
  src/test_node.cpp
  src/spacetime_vectorfield.cpp
  src/node.cpp
  src/window.cpp
  src/autonomous_particle.cpp
  src/line_shader.cpp 
  src/point_shader.cpp 
  src/renderable.cpp 
  src/scene.cpp 
  src/lic.cpp
  src/autonomous_particles_renderer.cpp
  src/pin.cpp

  src/imgui-node-editor/crude_json.cpp
  src/imgui-node-editor/imgui_canvas.cpp
  src/imgui-node-editor/imgui_node_editor_api.cpp
  src/imgui-node-editor/imgui_node_editor.cpp)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  target_compile_options(flowexplorer PRIVATE "-fcolor-diagnostics")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  target_compile_options(flowexplorer PRIVATE  "-fdiagnostics-color=always")
endif()
target_link_libraries(flowexplorer tatooine yavin)
target_compile_features(flowexplorer PUBLIC cxx_std_17)
target_compile_options(flowexplorer PRIVATE
  -Wall -Wextra -pedantic
  -Wno-tautological-undefined-compare -Wno-gnu-zero-variadic-macro-arguments)
target_include_directories(
  flowexplorer
  PUBLIC ${TATFE_INCLUDE_DIR} ${TATFE_BIN_INCLUDE_DIR})

add_custom_target(flowexplorer.run ./flowexplorer DEPENDS tatooine flowexplorer)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
set (FLOWEXPLORER_LINE_VERT ${TATFE_SHADER_DIR}/line/line.vert)
set (FLOWEXPLORER_LINE_FRAG ${TATFE_SHADER_DIR}/line/line.frag)
configure_file(${TATFE_INCLUDE_DIR}/tatooine/flowexplorer/line_shader.h.in
               ${TATFE_BIN_INCLUDE_DIR}/tatooine/flowexplorer/line_shader.h)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
set (FLOWEXPLORER_POINT_VERT ${TATFE_SHADER_DIR}/point/point.vert)
set (FLOWEXPLORER_POINT_FRAG ${TATFE_SHADER_DIR}/point/point.frag)
configure_file(${TATFE_INCLUDE_DIR}/tatooine/flowexplorer/point_shader.h.in
               ${TATFE_BIN_INCLUDE_DIR}/tatooine/flowexplorer/point_shader.h)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
set(TATOOINE_FLOWEXPLORER_SHADERS_AUTONOMOUS_PARTICLES_RENDERER_2D_VERT
    ${TATFE_SHADER_DIR}/autonomous_particles_renderer_2d.vert)
set(TATOOINE_FLOWEXPLORER_SHADERS_AUTONOMOUS_PARTICLES_RENDERER_2D_GEOM
    ${TATFE_SHADER_DIR}/autonomous_particles_renderer_2d.geom)
set(TATOOINE_FLOWEXPLORER_SHADERS_AUTONOMOUS_PARTICLES_RENDERER_2D_FRAG
    ${TATFE_SHADER_DIR}/autonomous_particles_renderer_2d.frag)
set(TATOOINE_FLOWEXPLORER_SHADERS_AUTONOMOUS_PARTICLES_RENDERER_3D_VERT
    ${TATFE_SHADER_DIR}/autonomous_particles_renderer_3d.vert)
set(TATOOINE_FLOWEXPLORER_SHADERS_AUTONOMOUS_PARTICLES_RENDERER_3D_GEOM
    ${TATFE_SHADER_DIR}/autonomous_particles_renderer_3d.geom)
set(TATOOINE_FLOWEXPLORER_SHADERS_AUTONOMOUS_PARTICLES_RENDERER_3D_FRAG
    ${TATFE_SHADER_DIR}/autonomous_particles_renderer_3d.frag)
configure_file(${TATFE_INCLUDE_DIR}/tatooine/flowexplorer/nodes/autonomous_particles_renderer.h.in
               ${TATFE_BIN_INCLUDE_DIR}/tatooine/flowexplorer/nodes/autonomous_particles_renderer.h)
