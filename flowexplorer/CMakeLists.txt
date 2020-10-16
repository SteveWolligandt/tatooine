if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  list(APPEND CMAKE_CXX_FLAGS "-fcolor-diagnostics")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  list(APPEND CMAKE_CXX_FLAGS "-fdiagnostics-color=always")
endif()
set(TATFE_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(TATFE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(TATFE_BIN_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
set(TATFE_SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
add_executable(
  flowexplorer

  src/flowexplorer.cpp
  src/node.cpp
  src/window.cpp
  #src/autonomous_particle.cpp
  src/line_shader.cpp 
  src/point_shader.cpp 
  src/renderable.cpp 
  src/scene.cpp 

  src/imgui-node-editor/crude_json.cpp
  src/imgui-node-editor/imgui_canvas.cpp
  src/imgui-node-editor/imgui_node_editor_api.cpp
  src/imgui-node-editor/imgui_node_editor.cpp)

add_custom_target(flowexplorer.run ./flowexplorer DEPENDS flowexplorer)
target_link_libraries(flowexplorer tatooine yavin)
target_compile_features(flowexplorer PUBLIC cxx_std_17)
target_include_directories(
  flowexplorer
  PUBLIC ${TATFE_INCLUDE_DIR} ${TATFE_BIN_INCLUDE_DIR})
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
