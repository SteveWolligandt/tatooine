add_library(rendering INTERFACE)
target_include_directories(rendering INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/include
  )
if (TATOOINE_GL_AVAILABLE)
  target_link_libraries(rendering INTERFACE gl geometry)
else()
  target_link_libraries(rendering INTERFACE geometry)
endif()
#-------------------------------------------------------------------------------
if (TATOOINE_GL_AVAILABLE)
#-------------------------------------------------------------------------------
set(TATOOINE_LIC_COMP_SHADER
    ${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/gpu/lic.comp)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/gpu/lic.h.in"
               "${CMAKE_CURRENT_BINARY_DIR}/include/tatooine/gpu/lic.h")

## Line Shader
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/gpu/line_shader.h.in"
               "${CMAKE_CURRENT_BINARY_DIR}/include/tatooine/gpu/line_shader.h")

## Texture Shader
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/gpu/texture_shader.h.in"
               "${CMAKE_CURRENT_BINARY_DIR}/include/tatooine/gpu/texture_shader.h")

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/gpu/reduce.h.in"
               "${CMAKE_CURRENT_BINARY_DIR}/include/tatooine/gpu/reduce.h")
#-------------------------------------------------------------------------------
endif()
#-------------------------------------------------------------------------------
# Get geometry properties for tatooineConfig.cmake
# include directories
get_target_property(RENDERING_INCLUDE_DIRECTORIES rendering INTERFACE_INCLUDE_DIRECTORIES)
foreach(INCLUDE_DIRECTORY ${RENDERING_INCLUDE_DIRECTORIES})
  if(${INCLUDE_DIRECTORY} MATCHES ".*BUILD_INTERFACE:.*")
    string(LENGTH ${INCLUDE_DIRECTORY} len)
    math(EXPR len ${len}-19)
    string(SUBSTRING ${INCLUDE_DIRECTORY} 18 ${len} INCLUDE_DIRECTORY)
    list(APPEND FILTERED_RENDERING_INCLUDE_DIRECTORIES ${INCLUDE_DIRECTORY})
  elseif(NOT (INCLUDE_DIRECTORY MATCHES "\<.*_INTERFACE:.*\>"))
    list(APPEND FILTERED_RENDERING_INCLUDE_DIRECTORIES ${INCLUDE_DIRECTORY})
  else()
  endif()
endforeach()
set(RENDERING_INCLUDE_DIRECTORIES ${FILTERED_RENDERING_INCLUDE_DIRECTORIES} PARENT_SCOPE)

get_target_property(RENDERING_LINK_LIBRARIES rendering INTERFACE_LINK_LIBRARIES)
foreach(LINK_LIBRARY ${RENDERING_LINK_LIBRARIES})
  if(${LINK_LIBRARY} MATCHES "gl")
    list(APPEND FILTERED_RENDERING_LINK_LIBRARIES "tatooine::gl")
  elseif(${LINK_LIBRARY} MATCHES "geometry")
    list(APPEND FILTERED_RENDERING_LINK_LIBRARIES "tatooine::geometry")
  endif()
endforeach()
set(RENDERING_LINK_LIBRARIES ${FILTERED_RENDERING_LINK_LIBRARIES} PARENT_SCOPE)

get_target_property(RENDERING_COMPILE_OPTIONS rendering INTERFACE_COMPILE_OPTIONS)
set(RENDERING_COMPILE_OPTIONS ${RENDERING_COMPILE_OPTIONS} PARENT_SCOPE)

if (TATOOINE_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
