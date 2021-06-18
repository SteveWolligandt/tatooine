#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# find OpenGL packages
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
set(OpenGL_GL_PREFERENCE "GLVND")
find_package(OpenGL COMPONENTS OpenGL EGL)
if (OpenGL_FOUND)
  set(TATOOINE_OPENGL_AVAILABLE 1)
else()
  set(TATOOINE_OPENGL_AVAILABLE 0)
endif()
if (TATOOINE_OPENGL_AVAILABLE) 
  option(TATOOINE_BUILD_GL "build OpenGL wrapper" OFF)
endif()
if (TATOOINE_BUILD_GL)
  set(TATOOINE_RENDERING_GL_AVAILABLE 1)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# GLFW
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
#set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
update_git_submodule(external/glfw)
add_subdirectory(external/glfw)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# build imgui
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
set(TATOOINE_RENDERING_GL_IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui_node_editor_version)
set(TATOOINE_RENDERING_GL_IMGUI_INCLUDE_DIRS ${TATOOINE_RENDERING_GL_IMGUI_DIR})
set(TATOOINE_RENDERING_GL_IMGUI_FILES
  src/glad_egl.c
  src/glad.c
  ${TATOOINE_RENDERING_GL_IMGUI_DIR}/imgui.cpp
  ${TATOOINE_RENDERING_GL_IMGUI_DIR}/imgui_widgets.cpp
  ${TATOOINE_RENDERING_GL_IMGUI_DIR}/imgui_draw.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/misc/cpp/imgui_stdlib.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/misc/fonts/binary_to_compressed_c.cpp
  )


#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# source files
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
set(TATOOINE_RENDERING_GL_SOURCES
  ${CMAKE_SOURCE_DIR}/src/atomiccounterbuffer.cpp
  ${CMAKE_SOURCE_DIR}/src/computeshader.cpp
  ${CMAKE_SOURCE_DIR}/src/errorcheck.cpp
  ${CMAKE_SOURCE_DIR}/src/fragmentshader.cpp
  ${CMAKE_SOURCE_DIR}/src/framebuffer.cpp
  ${CMAKE_SOURCE_DIR}/src/geometryshader.cpp
  ${CMAKE_SOURCE_DIR}/src/glfunctions.cpp
  ${CMAKE_SOURCE_DIR}/src/glslvar.cpp
  ${CMAKE_SOURCE_DIR}/src/glwrapper.cpp
  ${CMAKE_SOURCE_DIR}/src/indexbuffer.cpp
  ${CMAKE_SOURCE_DIR}/src/mutexhandler.cpp
  ${CMAKE_SOURCE_DIR}/src/shader.cpp
  ${CMAKE_SOURCE_DIR}/src/shaderstage.cpp
  ${CMAKE_SOURCE_DIR}/src/shaderstageparser.cpp
  ${CMAKE_SOURCE_DIR}/src/tesselationcontrolshader.cpp
  ${CMAKE_SOURCE_DIR}/src/tesselationevaluationshader.cpp
  ${CMAKE_SOURCE_DIR}/src/texture.cpp
  ${CMAKE_SOURCE_DIR}/src/vertexarray.cpp
  ${CMAKE_SOURCE_DIR}/src/vertexshader.cpp
  ${CMAKE_SOURCE_DIR}/src/window_notifier.cpp
  ${CMAKE_SOURCE_DIR}/src/keyboard.cpp
  ${CMAKE_SOURCE_DIR}/src/mouse.cpp
  ${CMAKE_SOURCE_DIR}/src/imgui_api_backend.cpp
  ${CMAKE_SOURCE_DIR}/src/imgui_shader.cpp
  ${CMAKE_SOURCE_DIR}/src/imgui_render_backend.cpp
  ${CMAKE_SOURCE_DIR}/src/imgui.cpp
  ${CMAKE_SOURCE_DIR}/src/glfw/api.cpp
  ${CMAKE_SOURCE_DIR}/src/glfw/base.cpp
  ${CMAKE_SOURCE_DIR}/src/glfw/context.cpp
  ${CMAKE_SOURCE_DIR}/src/glfw/window.cpp
  ${CMAKE_BINARY_DIR}/src/glverbose.cpp
  ${CMAKE_SOURCE_DIR}/src/context.cpp
  ${CMAKE_SOURCE_DIR}/src/window.cpp
)
add_library(rendering.gl SHARED ${TATOOINE_RENDERING_GL_SOURCES} ${TATOOINE_RENDERING_GL_IMGUI_FILES})
target_compile_features(rendering.gl PUBLIC cxx_std_20)

target_compile_definitions(rendering.gl PRIVATE DLL_EXPORT)
target_compile_options(rendering.gl PRIVATE
                       -Wall -Wextra -pedantic)

target_include_directories(rendering.gl
  PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/single_include
    ${CMAKE_CURRENT_BINARY_DIR}/include
    ${OPENGL_INCLUDE_DIR} 
    ${TATOOINE_RENDERING_GL_IMGUI_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui-filebrowser)

target_link_libraries(
  rendering.gl
   PUBLIC
    tatooine
   PRIVATE
    OpenGL::OpenGL
    OpenGL::EGL
    glfw)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# OpenGL call output
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
option(TATOOINE_RENDERING_GL_VERBOSE "verbose OpenGL calls" OFF)
if (TATOOINE_RENDERING_GL_VERBOSE)
  set(VERBOSE_GL "true")
else()
  set(VERBOSE_GL "false")
endif()

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# configure files
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
configure_file(src/glverbose.cpp.in
               src/glverbose.cpp)
configure_file(include/tatooine/rendering/gl/texpng.h.in
               include/tatooine/rendering/gl/texpng.h)
else()
  set(TATOOINE_RENDERING_GL_AVAILABLE 0)
endif()
target_compile_definitions(tatooine PUBLIC TATOOINE_RENDERING_GL_AVAILABLE=${TATOOINE_RENDERING_GL_AVAILABLE})
