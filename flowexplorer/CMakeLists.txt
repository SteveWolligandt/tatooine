update_git_submodule(external/tomlplusplus)
update_git_submodule(external/imgui-node-editor)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
set(TATFE_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(TATFE_BIN_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR}/src)
set(TATFE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(TATFE_BIN_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
set(TATFE_SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
set(TATOOINE_FLOWEXPLORER_FONTS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/fonts)
set(TATOOINE_FLOWEXPLORER_ICONS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/icons)
configure_file(${TATFE_SRC_DIR}/directories.cpp.in
               ${TATFE_BIN_SRC_DIR}/directories.cpp)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
add_executable(
  flowexplorer
  #${TATFE_SRC_DIR}/jacobian.cpp
  #${TATFE_SRC_DIR}/vtk_line_reader.cpp
  #${TATFE_SRC_DIR}/parallel_vectors.cpp
  ${TATFE_SRC_DIR}/sample_to_grid.cpp
  ${TATFE_SRC_DIR}/rectilinear_grid_vertex_property_renderer.cpp
  ${TATFE_SRC_DIR}/linspace.cpp
  #${TATFE_SRC_DIR}/random_points.cpp
  ${TATFE_SRC_DIR}/rectilinear_grid.cpp
  #${TATFE_SRC_DIR}/distance_measure.cpp
  #${TATFE_SRC_DIR}/unary_scalar_operation.cpp
  ${TATFE_SRC_DIR}/binary_operation.cpp
  #${TATFE_SRC_DIR}/sphere.cpp
  ${TATFE_SRC_DIR}/scalar.cpp
  #${TATFE_SRC_DIR}/vectorfield_to_gpu.cpp
  #${TATFE_SRC_DIR}/spacetime_splitted_vectorfield.cpp
  #${TATFE_SRC_DIR}/field_from_file.cpp
  #${TATFE_SRC_DIR}/axis_aligned_bounding_box.cpp
  #${TATFE_SRC_DIR}/iECTILINEAR_position.cpp
  #${TATFE_SRC_DIR}/RECTILINEAR_duffing_oscillator.cpp
  #${TATFE_SRC_DIR}/pathline.cpp
  #${TATFE_SRC_DIR}/random_pathlines.cpp
  #${TATFE_SRC_DIR}/spacetime_vectorfield.cpp
  #${TATFE_SRC_DIR}/lic.cpp

  ${TATFE_SRC_DIR}/doublegyre.cpp
  #${TATFE_SRC_DIR}/doublegyre3d.cpp
  #${TATFE_SRC_DIR}/abcflow.cpp
  #${TATFE_SRC_DIR}/saddle.cpp
  #${TATFE_SRC_DIR}/rayleigh_benard_convection.cpp

  #${TATFE_SRC_DIR}/autonomous_particles_flowmap.cpp
  #${TATFE_SRC_DIR}/autonomous_particles_flowmap_evaluator.cpp
  #${TATFE_SRC_DIR}/autonomous_particles_renderer.cpp
  #${TATFE_SRC_DIR}/autonomous_particle.cpp

  ${TATFE_SRC_DIR}/line_shader.cpp 
  ${TATFE_SRC_DIR}/point_shader.cpp 

  ${TATFE_SRC_DIR}/flowexplorer.cpp
  ${TATFE_BIN_SRC_DIR}/directories.cpp
  ${TATFE_SRC_DIR}/test_node.cpp
  ${TATFE_SRC_DIR}/window.cpp
  ${TATFE_SRC_DIR}/draw_icon.cpp
  ${TATFE_SRC_DIR}/node.cpp
  ${TATFE_SRC_DIR}/node_builder.cpp
  ${TATFE_SRC_DIR}/renderable.cpp 
  ${TATFE_SRC_DIR}/scene.cpp 
  ${TATFE_SRC_DIR}/pin.cpp

  external/imgui-node-editor/crude_json.cpp
  external/imgui-node-editor/imgui_canvas.cpp
  external/imgui-node-editor/imgui_node_editor_api.cpp
  external/imgui-node-editor/imgui_node_editor.cpp)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  target_compile_options(flowexplorer PRIVATE "-fcolor-diagnostics")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  target_compile_options(flowexplorer PRIVATE  "-fdiagnostics-color=always")
endif()
target_link_libraries(flowexplorer tatooine gl)
target_compile_features(flowexplorer PUBLIC cxx_std_17)
target_include_directories(flowexplorer
 PUBLIC
  ${TATFE_INCLUDE_DIR} ${TATFE_BIN_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/external/tomlplusplus/include
  ${CMAKE_CURRENT_SOURCE_DIR}/external)


if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR
    "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  target_compile_options(tatooine PUBLIC
    -Wno-gnu-zero-variadic-macro-arguments)
endif()

add_custom_target(flowexplorer.run ./flowexplorer DEPENDS tatooine flowexplorer)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
set (FLOWEXPLORER_SPHERE_VERT ${TATFE_SHADER_DIR}/sphere/sphere.vert)
set (FLOWEXPLORER_SPHERE_FRAG ${TATFE_SHADER_DIR}/sphere/sphere.frag)
configure_file(${TATFE_INCLUDE_DIR}/tatooine/flowexplorer/nodes/sphere.h.in
               ${TATFE_BIN_INCLUDE_DIR}/tatooine/flowexplorer/nodes/sphere.h)
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
