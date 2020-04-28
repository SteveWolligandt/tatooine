#===============================================================================
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "-fcolor-diagnostics")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "-fdiagnostics-color=always")
endif()
#===============================================================================
set(DATASET_DIR
  "/home/steve/vectorfield_datasets" CACHE PATH "Path to datasets")
add_compile_definitions(DATASET_DIR="${DATASET_DIR}")

function (copy FILE)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/${FILE} 
                 ${CMAKE_CURRENT_BINARY_DIR}/${FILE} COPYONLY)
endfunction ()
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/sync.sh.in
               ${CMAKE_CURRENT_BINARY_DIR}/sync.sh)

set(TATOOINE_STEADIFICATION_LIC_COMP
    ${CMAKE_CURRENT_SOURCE_DIR}/lic.comp)
set(TATOOINE_STEADIFICATION_SSF_RASTERIZATION_VERT
    ${CMAKE_CURRENT_SOURCE_DIR}/ssf_rasterization.vert)
set(TATOOINE_STEADIFICATION_SSF_RASTERIZATION_FRAG
    ${CMAKE_CURRENT_SOURCE_DIR}/ssf_rasterization.frag)
set(TATOOINE_STEADIFICATION_TEX_RASTERIZATION_TO_BUFFER_COMP
    ${CMAKE_CURRENT_SOURCE_DIR}/tex_rasterization_to_buffer.comp)
set(TATOOINE_STEADIFICATION_LL_TO_V_TEX_COMP
    ${CMAKE_CURRENT_SOURCE_DIR}/ll_to_v_tex.comp)
set(TATOOINE_STEADIFICATION_LL_TO_CURV_TEX_COMP
    ${CMAKE_CURRENT_SOURCE_DIR}/ll_to_curvature_tex.comp)
  set(TATOOINE_STEADIFICATION_SEEDCURVE_VERT
    ${CMAKE_CURRENT_SOURCE_DIR}/seedcurve.vert)
  set(TATOOINE_STEADIFICATION_SEEDCURVE_FRAG
    ${CMAKE_CURRENT_SOURCE_DIR}/seedcurve.frag)
set(TATOOINE_STEADIFICATION_FRAGMENT_COUNT_FRAG
    ${CMAKE_CURRENT_SOURCE_DIR}/fragment_count.frag)
set(TATOOINE_STEADIFICATION_WEIGHT_COMP
    ${CMAKE_CURRENT_SOURCE_DIR}/weight.comp)
set(TATOOINE_STEADIFICATION_COMBINE_RASTERIZATIONS_COMP
    ${CMAKE_CURRENT_SOURCE_DIR}/combine_rasterizations.comp)
set(TATOOINE_STEADIFICATION_COVERAGE_COMP
    ${CMAKE_CURRENT_SOURCE_DIR}/coverage.comp)
set(TATOOINE_STEADIFICATION_DUAL_COVERAGE_COMP
    ${CMAKE_CURRENT_SOURCE_DIR}/dual_coverage.comp)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/shaders.h.in"
               "${CMAKE_CURRENT_BINARY_DIR}/shaders.h")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/viridis_color_scale.png 
  ${CMAKE_CURRENT_BINARY_DIR}/color_scale.png COPYONLY)

set(HDF5_LIBS -lhdf5 -lhdf5_cpp -lhdf5_hl)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
add_executable (steadification main.cpp renderers.cpp shaders.cpp datasets.cpp)
target_include_directories(steadification PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(steadification tatooine yavin)
target_compile_features(steadification PUBLIC cxx_std_17)
target_compile_features(steadification PUBLIC cxx_std_17)
target_compile_options(steadification PRIVATE
  -Wall -Wextra -pedantic -Wno-missing-braces -Wno-unused-lambda-capture
  -march=native)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
add_executable (steadification.batch batch.cpp renderers.cpp shaders.cpp datasets.cpp)
target_include_directories(steadification.batch PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(steadification.batch tatooine yavin)
target_compile_features(steadification.batch PUBLIC cxx_std_17)
target_compile_features(steadification.batch PUBLIC cxx_std_17)
target_compile_options(steadification.batch PRIVATE
  -Wall -Wextra -pedantic -Wno-missing-braces -Wno-unused-lambda-capture
  -march=native)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
if(CLANG_TIDY_EXECUTABLE)
  set(TATOOINE_CLANG_TIDY_CMD
    ${CLANG_TIDY_EXECUTABLE}
    ${TATOOINE_CLANG_TIDY_CHECKS}
    "-header-filter=steadification")
endif()
if(CLANG_TIDY_EXECUTABLE AND TATOOINE_USE_CLANG_TIDY)
  set_target_properties(
    steadification PROPERTIES CXX_CLANG_TIDY ${TATOOINE_CLANG_TIDY_CMD})
endif()
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
add_executable (integrate_grid_edge integrate_grid_edge.cpp)
target_include_directories(integrate_grid_edge PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(integrate_grid_edge tatooine)
target_compile_features(integrate_grid_edge PUBLIC cxx_std_17)
target_compile_features(integrate_grid_edge PUBLIC cxx_std_17)
target_compile_options(integrate_grid_edge PRIVATE
  -Wall -Wextra -pedantic -Wno-missing-braces -Wno-unused-lambda-capture
  -march=native)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
add_executable (read_vtk_test read_vtk_test.cpp)
target_include_directories(read_vtk_test PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(read_vtk_test tatooine)
target_compile_features(read_vtk_test PUBLIC cxx_std_17)
target_compile_features(read_vtk_test PUBLIC cxx_std_17)
target_compile_options(read_vtk_test PRIVATE
  -Wall -Wextra -pedantic -Wno-missing-braces -Wno-unused-lambda-capture
  -march=native)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
set(TATOOINE_STEADIFICATION_USE_FAST_MATH false CACHE BOOL
    "use -ffast-math for steadification")
option(TATOOINE_STEADIFICATION_USE_FAST_MATH
    "use -ffast-math for steadification" OFF)
if (TATOOINE_STEADIFICATION_USE_FAST_MATH)
  target_compile_options(steadification PRIVATE -ffast-math)
endif()
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
add_custom_target(steadification.sync
                  COMMAND chmod +x ${CMAKE_CURRENT_BINARY_DIR}/sync.sh
                  COMMAND ${CMAKE_CURRENT_BINARY_DIR}/sync.sh
                  USES_TERMINAL)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
add_custom_target(steadification.doublegyre
                  COMMAND ./steadification dg -5 5 2 0.01 0.99
                  DEPENDS steadification)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
add_executable (boussinesq_test boussinesq_test.cpp datasets.cpp)
target_include_directories(boussinesq_test PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(boussinesq_test tatooine yavin)
target_compile_features(boussinesq_test PUBLIC cxx_std_17)
target_compile_options(boussinesq_test PRIVATE
  -Wall -Wextra -pedantic -Wno-missing-braces -Wno-unused-lambda-capture)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
set(OpenGL_GL_PREFERENCE "GLVND")
find_package(OpenGL REQUIRED)
find_package(GLUT)
add_executable(ibfv ibfv.cpp)
target_include_directories(ibfv PUBLIC ${GLUT_INCLUDE_DIRS})
target_link_libraries(ibfv OpenGL::OpenGL OpenGL::GLU ${GLUT_LIBRARY})
target_compile_features(ibfv PUBLIC cxx_std_17)
add_custom_target(ibfv.run.dg 
                  COMMAND ./ibfv dg 0 1000 0.05
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                  DEPENDS ibfv)
