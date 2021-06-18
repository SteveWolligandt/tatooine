#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# find OpenGL packages
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
set(OpenGL_GL_PREFERENCE "GLVND")
find_package(OpenGL COMPONENTS OpenGL EGL)
if (OpenGL_FOUND) 
  option(TATOOINE_BUILD_GL "build OpenGL wrapper" OFF)
if (TATOOINE_BUILD_GL)
  set(TATOOINE_RENDERING_GL_AVAILABLE 1 PARENT_SCOPE)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# GLFW
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
#set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
update_git_submodule(external/glfw)
update_git_submodule(external/imgui)
update_git_submodule(external/imgui-filebrowser)
add_subdirectory(external/glfw)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# build imgui
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
set(TATOOINE_RENDERING_GL_IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui)
set(TATOOINE_RENDERING_GL_IMGUI_INCLUDE_DIRS ${TATOOINE_RENDERING_GL_IMGUI_DIR})
set(TATOOINE_RENDERING_GL_IMGUI_FILES
  src/glad_egl.c
  src/glad.c
  ${TATOOINE_RENDERING_GL_IMGUI_DIR}/imgui.cpp
  ${TATOOINE_RENDERING_GL_IMGUI_DIR}/imgui_widgets.cpp
  ${TATOOINE_RENDERING_GL_IMGUI_DIR}/imgui_draw.cpp
  ${TATOOINE_RENDERING_GL_IMGUI_DIR}/imgui_tables.cpp
  ${TATOOINE_RENDERING_GL_IMGUI_DIR}/misc/cpp/imgui_stdlib.cpp
  ${TATOOINE_RENDERING_GL_IMGUI_DIR}/misc/fonts/binary_to_compressed_c.cpp
  )


#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# source files
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
set(TATOOINE_RENDERING_GL_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/atomiccounterbuffer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/computeshader.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/errorcheck.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/fragmentshader.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/framebuffer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/geometryshader.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/glfunctions.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/glslvar.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/glwrapper.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/indexbuffer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/mutexhandler.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/shader.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/shaderstage.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/shaderstageparser.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/tesselationcontrolshader.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/tesselationevaluationshader.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/texture.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/vertexarray.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/vertexshader.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/window_notifier.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/keyboard.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/mouse.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/imgui_api_backend.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/imgui_shader.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/imgui_render_backend.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/imgui.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/glfw/api.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/glfw/base.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/glfw/context.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/glfw/window.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/context.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/window.cpp
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
  target_compile_definitions(rendering.gl PUBLIC TATOOINE_RENDERING_GL_VERBOSE=1)
else()
  target_compile_definitions(rendering.gl PUBLIC TATOOINE_RENDERING_GL_VERBOSE=0)
endif()

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# configure files
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/rendering/gl/texpng.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/include/tatooine/rendering/gl/texpng.h)
else()
  set(TATOOINE_RENDERING_GL_AVAILABLE 0 PARENT_SCOPE)
endif()
else()
  set(TATOOINE_RENDERING_GL_AVAILABLE 0 PARENT_SCOPE)
endif()
target_compile_definitions(tatooine PUBLIC TATOOINE_RENDERING_GL_AVAILABLE=${TATOOINE_RENDERING_GL_AVAILABLE})
