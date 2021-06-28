#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# find OpenGL packages
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
set(OpenGL_GL_PREFERENCE "GLVND")
find_package(OpenGL COMPONENTS OpenGL EGL)
if (OpenGL_FOUND) 
  option(TATOOINE_BUILD_GL "build OpenGL wrapper" ON)
if (TATOOINE_BUILD_GL)
  set(TATOOINE_GL_AVAILABLE 1 PARENT_SCOPE)
  message(STATUS "OpenGL available")
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
# build glad
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
add_library(glad STATIC external/glad/glad_egl.c external/glad/glad.c)
target_include_directories(glad PUBLIC external/glad)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# build imgui
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
set(TATOOINE_GL_IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui)
set(TATOOINE_GL_IMGUI_INCLUDE_DIRS ${TATOOINE_GL_IMGUI_DIR})
set(TATOOINE_GL_IMGUI_FILES
  ${TATOOINE_GL_IMGUI_DIR}/imgui.cpp
  ${TATOOINE_GL_IMGUI_DIR}/imgui_widgets.cpp
  ${TATOOINE_GL_IMGUI_DIR}/imgui_draw.cpp
  ${TATOOINE_GL_IMGUI_DIR}/imgui_tables.cpp
  ${TATOOINE_GL_IMGUI_DIR}/misc/cpp/imgui_stdlib.cpp
  ${TATOOINE_GL_IMGUI_DIR}/misc/fonts/binary_to_compressed_c.cpp
  )
add_library(imgui STATIC ${TATOOINE_GL_IMGUI_FILES})
target_include_directories(imgui PUBLIC ${TATOOINE_GL_IMGUI_DIR})

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# source files
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
set(TATOOINE_GL_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/atomiccounterbuffer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/bezier_widget.cpp
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
add_library(gl STATIC ${TATOOINE_GL_SOURCES})
target_compile_definitions(gl PUBLIC TATOOINE_GL_AVAILABLE=${TATOOINE_GL_AVAILABLE})
target_compile_features(gl PUBLIC cxx_std_20)
target_compile_definitions(gl PRIVATE DLL_EXPORT)
target_compile_options(gl PRIVATE
                       -Wall -Wextra -pedantic)

target_include_directories(gl
  PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/single_include
    ${CMAKE_CURRENT_BINARY_DIR}/include
    ${OPENGL_INCLUDE_DIR} 
    ${TATOOINE_GL_IMGUI_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui
    ${CMAKE_CURRENT_SOURCE_DIR}/external/imgui-filebrowser)

target_link_libraries(
  gl
   PUBLIC
    tatooine
    glad
   PRIVATE
    OpenGL::OpenGL
    OpenGL::EGL
    glfw
    imgui)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# OpenGL call output
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
option(TATOOINE_GL_VERBOSE "verbose OpenGL calls" OFF)
if (TATOOINE_GL_VERBOSE)
  target_compile_definitions(gl PUBLIC TATOOINE_GL_VERBOSE=1)
else()
  target_compile_definitions(gl PUBLIC TATOOINE_GL_VERBOSE=0)
endif()

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# configure files
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/tatooine/gl/texpng.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/include/tatooine/gl/texpng.h)
else()
  set(TATOOINE_GL_AVAILABLE 0 PARENT_SCOPE)
endif()
else()
  set(TATOOINE_GL_AVAILABLE 0 PARENT_SCOPE)
endif()
