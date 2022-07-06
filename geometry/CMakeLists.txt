list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/")
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
target_link_libraries(geometry PUBLIC tensor fields)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# CDT
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
option(TATOOINE_USE_CDT "By setting this to ON tatooine will checkout CDT from github and compile it." ON)
if (TATOOINE_USE_CDT)
  update_git_submodule(external/CDT)
  set(TATOOINE_CDT_AVAILABLE 1)
  set(CDT_USE_AS_COMPILED_LIBRARY ON CACHE BOOL "If enabled templates for float and double will be instantiated and compiled into a library")
  set(CDT_USE_BOOST ON CACHE BOOL "If enabled Boost::rtree can be used for finding the nearest point; boost::flat_set is used for better performance")
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/CDT/CDT)
  target_link_libraries(geometry PUBLIC CDT)
else()
  set(TATOOINE_CDT_AVAILABLE 0)
endif()
set(TATOOINE_CDT_AVAILABLE ${TATOOINE_CDT_AVAILABLE} PARENT_SCOPE)
target_compile_definitions(geometry PUBLIC TATOOINE_CDT_AVAILABLE=${TATOOINE_CDT_AVAILABLE})
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# FLANN
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
find_package(Flann)
if (Flann_FOUND)
  find_library(LZ4_LIBRARIES NAMES liblz4.so HINTS "lz4/lib")
  set(TATOOINE_FLANN_AVAILABLE 1)
  target_link_libraries(geometry PUBLIC ${LZ4_LIBRARIES})
  target_link_libraries(geometry PUBLIC flann)
else()
  set(TATOOINE_FLANN_AVAILABLE 0)
endif()
set(TATOOINE_FLANN_AVAILABLE ${TATOOINE_FLANN_AVAILABLE} PARENT_SCOPE)
target_compile_definitions(geometry PUBLIC TATOOINE_FLANN_AVAILABLE=${TATOOINE_FLANN_AVAILABLE})
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# CGAL
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
set(CGAL_DO_NOT_WARN_ABOUT_CMAKE_BUILD_TYPE TRUE PARENT_SCOPE)
set(CGAL_DATA_DIR ${CMAKE_CURRENT_BINARY_DIR}/cgal_data_dir)
find_package(CGAL QUIET)

if (CGAL_FOUND)
  target_link_libraries(geometry PUBLIC CGAL::CGAL)
  target_compile_definitions(geometry PUBLIC -DCGAL_DISABLE_ROUNDING_MATH_CHECK)
  set(TATOOINE_CGAL_AVAILABLE 1)
  set(TATOOINE_USE_INSTALLED_CGAL 1)
  set(TATOOINE_USE_INSTALLED_CGAL 1 PARENT_SCOPE)
else()
  find_package(MPFR )
  find_package(GMP )
  if (MPFR_FOUND AND GMP_FOUND)
    update_git_submodule(external/cgal)
    target_link_libraries(geometry PUBLIC mpfr gmp)
    target_include_directories(geometry PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/external/cgal/Triangulation/include)
    target_compile_definitions(geometry PUBLIC -DCGAL_DISABLE_ROUNDING_MATH_CHECK)
    set(TATOOINE_CGAL_AVAILABLE 1)
    set(TATOOINE_USE_INSTALLED_CGAL 0)
    set(TATOOINE_USE_INSTALLED_CGAL 0 PARENT_SCOPE)
  else()
    set(TATOOINE_CGAL_AVAILABLE 0)
  endif()
endif()
set(TATOOINE_CGAL_AVAILABLE ${TATOOINE_CGAL_AVAILABLE})
set(TATOOINE_CGAL_AVAILABLE ${TATOOINE_CGAL_AVAILABLE} PARENT_SCOPE)
target_compile_definitions(geometry PUBLIC TATOOINE_CGAL_AVAILABLE=${TATOOINE_CGAL_AVAILABLE})

if (TATOOINE_BUILD_EXAMPLES)
  add_subdirectory(examples EXCLUDE_FROM_ALL)
endif()


#-------------------------------------------------------------------------------
# Get geometry properties for tatooineConfig.cmake
# include directories
get_target_property(GEOMETRY_INCLUDE_DIRECTORIES geometry INCLUDE_DIRECTORIES)
foreach(INCLUDE_DIRECTORY ${GEOMETRY_INCLUDE_DIRECTORIES})
  if(${INCLUDE_DIRECTORY} MATCHES ".*BUILD_INTERFACE:.*")
    string(LENGTH ${INCLUDE_DIRECTORY} len)
    math(EXPR len ${len}-19)
    string(SUBSTRING ${INCLUDE_DIRECTORY} 18 ${len} INCLUDE_DIRECTORY)
    list(APPEND FILTERED_GEOMETRY_INCLUDE_DIRECTORIES ${INCLUDE_DIRECTORY})
  elseif(NOT (INCLUDE_DIRECTORY MATCHES "\<.*_INTERFACE:.*\>"))
    list(APPEND FILTERED_GEOMETRY_INCLUDE_DIRECTORIES ${INCLUDE_DIRECTORY})
  else()
  endif()
endforeach()
set(GEOMETRY_INCLUDE_DIRECTORIES ${FILTERED_GEOMETRY_INCLUDE_DIRECTORIES} PARENT_SCOPE)

get_target_property(GEOMETRY_LINK_LIBRARIES geometry LINK_LIBRARIES)
foreach(LINK_LIBRARY ${GEOMETRY_LINK_LIBRARIES})
  if(${LINK_LIBRARY} MATCHES "tensor")
    list(APPEND FILTERED_GEOMETRY_LINK_LIBRARIES "tatooine::tensor")
  elseif(${LINK_LIBRARY} MATCHES "fields")
    list(APPEND FILTERED_GEOMETRY_LINK_LIBRARIES "tatooine::fields")
  elseif(${LINK_LIBRARY} MATCHES "CDT")
    list(APPEND FILTERED_GEOMETRY_LINK_LIBRARIES "tatooine::CDT")
  elseif(NOT (LINK_LIBRARY MATCHES "\<.*_INTERFACE:.*\>"))
    list(APPEND FILTERED_GEOMETRY_LINK_LIBRARIES ${LINK_LIBRARY})
  else()
  endif()
endforeach()
set(GEOMETRY_LINK_LIBRARIES ${FILTERED_GEOMETRY_LINK_LIBRARIES} PARENT_SCOPE)

get_target_property(GEOMETRY_COMPILE_DEFINITIONS geometry COMPILE_DEFINITIONS)
set(GEOMETRY_COMPILE_DEFINITIONS ${GEOMETRY_COMPILE_DEFINITIONS} PARENT_SCOPE)

get_target_property(GEOMETRY_COMPILE_OPTIONS geometry COMPILE_OPTIONS)
set(GEOMETRY_COMPILE_OPTIONS ${GEOMETRY_COMPILE_OPTIONS} PARENT_SCOPE)
