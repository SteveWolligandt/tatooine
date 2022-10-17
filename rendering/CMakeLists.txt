add_library(rendering INTERFACE)
target_include_directories(
  rendering INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include
                      ${CMAKE_CURRENT_BINARY_DIR}/include)
if(TATOOINE_GL_AVAILABLE)
  target_link_libraries(rendering INTERFACE gl geometry)
else()
  target_link_libraries(rendering INTERFACE geometry)
endif()
# ------------------------------------------------------------------------------
if(TATOOINE_GL_AVAILABLE)
  set(TATOOINE_LIC_COMP_SHADER
      ${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/gpu/lic.comp)
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/gpu/lic.h.in"
                 "${CMAKE_CURRENT_BINARY_DIR}/include/tatooine/gpu/lic.h")

  # Line Shader
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/gpu/line_shader.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/tatooine/gpu/line_shader.h")

  # Texture Shader
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/gpu/texture_shader.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/tatooine/gpu/texture_shader.h")

  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/gpu/reduce.h.in"
                 "${CMAKE_CURRENT_BINARY_DIR}/include/tatooine/gpu/reduce.h")
endif()
# ------------------------------------------------------------------------------
if(TATOOINE_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
