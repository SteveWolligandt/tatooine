if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  list(APPEND CMAKE_CXX_FLAGS "-fcolor-diagnostics")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  list(APPEND CMAKE_CXX_FLAGS "-fdiagnostics-color=always")
endif()
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
add_executable(pathline_render_window.test pathline_render_window_test.cpp)
target_link_libraries(pathline_render_window.test tatooine yavin)
target_compile_features(pathline_render_window.test PUBLIC cxx_std_17)
add_custom_target(pathline_render_window.test.run
                   COMMAND ./pathline_render_window.test
                   DEPENDS pathline_render_window.test)
