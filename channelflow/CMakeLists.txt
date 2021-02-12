add_executable(channelflow_visualization channelflow_visualization.cpp)
target_link_libraries(channelflow_visualization PUBLIC tatooine)

add_custom_target(
  channelflow_visualization.run
  COMMAND ./channelflow_visualization
  DEPENDS channelflow_visualization)

# force coloring
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  target_compile_options(channelflow_visualization PRIVATE "-fcolor-diagnostics")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  target_compile_options(channelflow_visualization PRIVATE  "-fdiagnostics-color=always")
endif()
