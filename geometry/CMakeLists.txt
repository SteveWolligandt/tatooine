add_library(geometry INTERFACE)
target_include_directories(geometry INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  #${CMAKE_CURRENT_BINARY_DIR}/include
  )
target_link_libraries(geometry INTERFACE tensor)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# CDT
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#option(TATOOINE_USE_CDT "By setting this to ON tatooine will checkout <F3>CDT from github and compile it." ON)
if (TATOOINE_USE_CDT)
  set(TATOOINE_CDT_AVAILABLE 1 PARENT_SCOPE)
  update_git_submodule(external/CDT)
  set(CDT_USE_AS_COMPILED_LIBRARY ON CACHE BOOL "If enabled templates for float and double will be instantiated and compiled into a library")
  set(CDT_USE_BOOST ON CACHE BOOL "If enabled Boost::rtree can be used for finding the nearest point; boost::flat_set is used for better performance")
  add_subdirectory(external/CDT/CDT)
  target_link_libraries(geometry INTERFACE CDT)
  target_compile_definitions(geometry INTERFACE TATOOINE_CDT_AVAILABLE=1)
  list(APPEND TATOOINE_EXPORTING_TARGETS CDT)
else()
  set(TATOOINE_CDT_AVAILABLE 0 PARENT_SCOPE)
  target_compile_definitions(geometry INTERFACE TATOOINE_CDT_AVAILABLE=0)
endif()
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# FLANN
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
find_package(Flann REQUIRED)
if (Flann_FOUND)
  message(STATUS "Found flann")
  set(TATOOINE_FLANN_AVAILABLE 1 PARENT_SCOPE)
  target_link_libraries(geometry INTERFACE flann)
else()
  set(TATOOINE_FLANN_AVAILABLE 0 PARENT_SCOPE)
  message(FATAL_ERROR "not found flann")
endif()
target_compile_definitions(geometry INTERFACE TATOOINE_FLANN_AVAILABLE=${TATOOINE_FLANN_AVAILABLE})
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# CGAL
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
set(CGAL_DO_NOT_WARN_ABOUT_CMAKE_BUILD_TYPE TRUE)
find_package(CGAL QUIET)
if (CGAL_FOUND)
  set(TATOOINE_CGAL_AVAILABLE 1 PARENT_SCOPE)
#  target_compile_definitions(geometry INTERFACE TATOOINE_HAS_CGAL_SUPPORT)
#  target_link_libraries(geometry INTERFACE CGAL::CGAL)
#  message(STATUS "CGAL available")
#  target_compile_definitions(geometry INTERFACE -DCGAL_DISABLE_ROUNDING_MATH_CHECK)
else()
  set(TATOOINE_CGAL_AVAILABLE 0 PARENT_SCOPE)
endif()
target_compile_definitions(geometry INTERFACE TATOOINE_CGAL_AVAILABLE=${TATOOINE_CGAL_AVAILABLE})
