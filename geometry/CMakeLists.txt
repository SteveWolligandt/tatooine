set(SRC_FILES 
  src/vtk_legacy.cpp
  src/vtk/xml/reader.cpp
  src/vtk/xml/data_attribute.cpp
  src/vtk/xml/byte_order.cpp
  src/parse.cpp
)
add_library(geometry STATIC ${SRC_FILES})
target_include_directories(geometry PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  #${CMAKE_CURRENT_BINARY_DIR}/include
  )
target_link_libraries(geometry PUBLIC tensor)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# CDT
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
option(TATOOINE_USE_CDT "By setting this to ON tatooine will checkout CDT from github and compile it." ON)
if (TATOOINE_USE_CDT)
  update_git_submodule(external/CDT)
  set(TATOOINE_CDT_AVAILABLE 1 PARENT_SCOPE)
  set(CDT_USE_AS_COMPILED_LIBRARY ON CACHE BOOL "If enabled templates for float and double will be instantiated and compiled into a library")
  set(CDT_USE_BOOST ON CACHE BOOL "If enabled Boost::rtree can be used for finding the nearest point; boost::flat_set is used for better performance")
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/CDT/CDT)
  target_link_libraries(geometry PUBLIC CDT)
else()
  set(TATOOINE_CDT_AVAILABLE 0 PARENT_SCOPE)
endif()
target_compile_definitions(geometry PUBLIC TATOOINE_CDT_AVAILABLE=${TATOOINE_CDT_AVAILABLE})
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# FLANN
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
find_package(Flann REQUIRED)
if (Flann_FOUND)
  set(TATOOINE_FLANN_AVAILABLE 1 PARENT_SCOPE)
  target_link_libraries(geometry PUBLIC flann)
else()
  set(TATOOINE_FLANN_AVAILABLE 0 PARENT_SCOPE)
endif()
target_compile_definitions(geometry PUBLIC TATOOINE_FLANN_AVAILABLE=${TATOOINE_FLANN_AVAILABLE})
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# CGAL
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
set(CGAL_DO_NOT_WARN_ABOUT_CMAKE_BUILD_TYPE TRUE)
set(CGAL_DATA_DIR ${CMAKE_CURRENT_BINARY_DIR}/cgal_data_dir)
find_package(CGAL QUIET)
if (CGAL_FOUND)
  set(TATOOINE_CGAL_AVAILABLE 1 PARENT_SCOPE)
  target_link_libraries(geometry PUBLIC CGAL::CGAL)
  target_compile_definitions(geometry PUBLIC -DCGAL_DISABLE_ROUNDING_MATH_CHECK)
else()
  set(TATOOINE_CGAL_AVAILABLE 0 PARENT_SCOPE)
endif()
target_compile_definitions(geometry PUBLIC TATOOINE_CGAL_AVAILABLE=${TATOOINE_CGAL_AVAILABLE})
