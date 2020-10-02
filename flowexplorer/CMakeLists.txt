if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  list(APPEND CMAKE_CXX_FLAGS "-fcolor-diagnostics")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  list(APPEND CMAKE_CXX_FLAGS "-fdiagnostics-color=always")
endif()
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
add_executable(
  flowexplorer
  flowexplorer.cpp line_shader.cpp 
  imgui-node-editor/crude_json.cpp
  imgui-node-editor/imgui_canvas.cpp
  imgui-node-editor/imgui_node_editor_api.cpp
  imgui-node-editor/imgui_node_editor.cpp)
add_custom_target(flowexplorer.run
  ./flowexplorer DEPENDS flowexplorer)
target_link_libraries(flowexplorer tatooine yavin)
target_compile_features(flowexplorer PUBLIC cxx_std_17)
target_include_directories(flowexplorer
                            PRIVATE
                             ${CMAKE_CURRENT_SOURCE_DIR}
                             ${CMAKE_CURRENT_BINARY_DIR})
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
set (FLOWEXPLORER_LINE_VERT ${CMAKE_CURRENT_SOURCE_DIR}/line.vert)
set (FLOWEXPLORER_LINE_FRAG ${CMAKE_CURRENT_SOURCE_DIR}/line.frag)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/line_shader.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/line_shader.h)
