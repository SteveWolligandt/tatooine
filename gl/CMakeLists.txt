#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# find OpenGL packages
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
set(OpenGL_GL_PREFERENCE "GLVND")
if("${CMAKE_SYSTEM_NAME}" STREQUAL "Windows")
  find_package(OpenGL COMPONENTS OpenGL)
elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
  find_package(OpenGL COMPONENTS OpenGL EGL)
endif()
if (OpenGL_FOUND) 
  option(TATOOINE_BUILD_GL "build OpenGL wrapper" ON)
else()
  set(TATOOINE_BUILD_GL OFF)
  set(TATOOINE_GL_AVAILABLE 0 PARENT_SCOPE)
endif()

if (TATOOINE_BUILD_GL)
  update_git_submodule(external/glfw)
  update_git_submodule(external/imgui)
  update_git_submodule(external/imgui-filebrowser)
  set(TATOOINE_GL_AVAILABLE 1 PARENT_SCOPE)
  #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  # GLFW
  #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GLFW_INSTALL OFF CACHE BOOL "Generate installation target" FORCE)
  add_subdirectory(external/glfw)
  #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  # build glad
  #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  set(GLAD_FILES external/glad/src/glad.c)
  if("${CMAKE_SYSTEM_NAME}" STREQUAL "Windows")
    list(APPEND GLAD_FILES external/glad_wgl/src/glad_wgl.c)
  elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
    list(APPEND GLAD_FILES external/glad_wgl/src/glad_egl.c)
  endif()
  add_library(glad STATIC ${GLAD_FILES})
  target_link_libraries(glad PUBLIC ${CMAKE_DL_LIBS})
  target_include_directories(glad PUBLIC external/glad/include)
  if("${CMAKE_SYSTEM_NAME}" STREQUAL "Windows")
    target_include_directories(glad PUBLIC external/glad_wgl/include)
  elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
    target_include_directories(glad PUBLIC external/glad_egl/include)
  endif()
  get_target_property(GLAD_LINK_LIBRARIES glad LINK_LIBRARIES)
  #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  # build imgui
  #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
  set(TATOOINE_GL_IMGUI_DIR external/imgui)
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
    src/atomiccounterbuffer.cpp
    src/bezier_widget.cpp
    src/computeshader.cpp
    src/errorcheck.cpp
    src/fragmentshader.cpp
    src/framebuffer.cpp
    src/geometryshader.cpp
    src/glfunctions.cpp
    src/glslvar.cpp
    src/glwrapper.cpp
    src/indexbuffer.cpp
    src/mutexhandler.cpp
    src/shader.cpp
    src/shaderstage.cpp
    src/shaderstageparser.cpp
    src/tesselationcontrolshader.cpp
    src/tesselationevaluationshader.cpp
    src/texture.cpp
    src/vertexarray.cpp
    src/vertexshader.cpp
    src/window_notifier.cpp
    src/keyboard.cpp
    src/mouse.cpp
    src/imgui_api_backend.cpp
    src/imgui_shader.cpp
    src/imgui_render_backend.cpp
    src/imgui.cpp
    src/glfw/api.cpp
    src/glfw/base.cpp
    src/glfw/context.cpp
    src/glfw/window.cpp
    src/context.cpp
    src/window.cpp
  )
  add_library(gl STATIC ${TATOOINE_GL_SOURCES})
  set_property(TARGET gl PROPERTY CXX_STANDARD 20)
  list(APPEND TATOOINE_EXPORTING_TARGETS gl imgui glfw glad)
  set(TATOOINE_EXPORTING_TARGETS ${TATOOINE_EXPORTING_TARGETS} PARENT_SCOPE)
  target_compile_definitions(gl PRIVATE DLL_EXPORT)
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR
      "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_compile_options(gl PRIVATE
                           -Wall -Wextra -pedantic)
  endif()

  target_include_directories(gl
    PUBLIC 
      "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
      "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
      ${TATOOINE_GL_IMGUI_INCLUDE_DIRS}
      external/imgui
      external/imgui-filebrowser)

  target_link_libraries(
    gl
     PUBLIC
      tensor
      glad
      glfw
     PRIVATE
      imgui)
  if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
    target_link_libraries(OpenGL)
  elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
    target_link_libraries(OpenGL::OpenGL OpenGL::EGL)
  endif()

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

  # Get gl properties for tatooineConfig.cmake
  # include directories
  get_target_property(GL_INCLUDE_DIRECTORIES gl INCLUDE_DIRECTORIES)
  foreach(INCLUDE_DIRECTORY ${GL_INCLUDE_DIRECTORIES})
    if(${INCLUDE_DIRECTORY} MATCHES ".*BUILD_INTERFACE:.*")
      string(LENGTH ${INCLUDE_DIRECTORY} len)
      math(EXPR len ${len}-19)
      string(SUBSTRING ${INCLUDE_DIRECTORY} 18 ${len} INCLUDE_DIRECTORY)
      list(APPEND FILTERED_GL_INCLUDE_DIRECTORIES ${INCLUDE_DIRECTORY})
    elseif(NOT (INCLUDE_DIRECTORY MATCHES "\<.*_INTERFACE:.*\>"))
      list(APPEND FILTERED_GL_INCLUDE_DIRECTORIES ${INCLUDE_DIRECTORY})
    else()
    endif()
  endforeach()
  set(GL_INCLUDE_DIRECTORIES ${FILTERED_GL_INCLUDE_DIRECTORIES} PARENT_SCOPE)

  get_target_property(GL_LINK_LIBRARIES gl LINK_LIBRARIES)
  foreach(LINK_LIBRARY ${GL_LINK_LIBRARIES})
        if(${LINK_LIBRARY} MATCHES "glad")
      list(APPEND FILTERED_GL_LINK_LIBRARIES "tatooine::glad")
    elseif(${LINK_LIBRARY} MATCHES "glfw")
      list(APPEND FILTERED_GL_LINK_LIBRARIES "tatooine::glfw")
    elseif(${LINK_LIBRARY} MATCHES "geometry")
      list(APPEND FILTERED_GL_LINK_LIBRARIES "tatooine::geometry")
    elseif(${LINK_LIBRARY} MATCHES "imgui")
      list(APPEND FILTERED_GL_LINK_LIBRARIES "tatooine::imgui")
    elseif(${LINK_LIBRARY} MATCHES "tensor")
      list(APPEND FILTERED_GL_LINK_LIBRARIES "tatooine::tensor")
    elseif(NOT (LINK_LIBRARY MATCHES "\<.*_INTERFACE:.*\>"))
      list(APPEND FILTERED_GL_LINK_LIBRARIES ${LINK_LIBRARY})
    endif()
  endforeach()
  set(GL_LINK_LIBRARIES ${FILTERED_GL_LINK_LIBRARIES} PARENT_SCOPE)

  get_target_property(GL_COMPILE_OPTIONS gl COMPILE_OPTIONS)
  set(GL_COMPILE_OPTIONS ${GL_COMPILE_OPTIONS} PARENT_SCOPE)

  get_target_property(GL_COMPILE_FEATURES gl COMPILE_FEATURES)
  set(GL_COMPILE_FEATURES ${GL_COMPILE_FEATURES} PARENT_SCOPE)

  get_target_property(GL_COMPILE_DEFINITIONS gl COMPILE_DEFINITIONS)
  set(GL_COMPILE_DEFINITIONS ${GL_COMPILE_DEFINITIONS} PARENT_SCOPE)
else()
  set(TATOOINE_GL_AVAILABLE 0 PARENT_SCOPE)
endif()
