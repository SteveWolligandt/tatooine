list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/")
set(CURRENT_EXTERNAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external)
set(SRC_FILES 
  src/vtk_legacy.cpp
  src/vtk/xml/reader.cpp
  src/vtk/xml/data_attribute.cpp
  src/vtk/xml/byte_order.cpp
  src/parse.cpp
)
add_library(geometry STATIC ${SRC_FILES})
set_property(TARGET geometry PROPERTY CXX_STANDARD 20)
target_include_directories(geometry PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(geometry PUBLIC tensor fields Boost::boost)
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
target_compile_definitions(geometry PUBLIC
  TATOOINE_FLANN_AVAILABLE=${TATOOINE_FLANN_AVAILABLE})
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
else()
  if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
    find_package(MPFR QUIET)
    find_package(GMP QUIET)
    if (MPFR_FOUND AND GMP_FOUND)
      update_git_submodule(external/cgal)
      target_link_libraries(geometry PUBLIC mpfr gmp)
      target_include_directories(geometry PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/external/cgal/Triangulation/include)
      target_compile_definitions(geometry PUBLIC -DCGAL_DISABLE_ROUNDING_MATH_CHECK)
      set(TATOOINE_CGAL_AVAILABLE 1)
      set(TATOOINE_USE_INSTALLED_CGAL 0)
    else()
      message(FATAL_ERROR "Could not build with CGAL. MPFR and GMP could not be found.")
      set(TATOOINE_CGAL_AVAILABLE 0)
      set(TATOOINE_USE_INSTALLED_CGAL 0)
    endif()
  elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "Windows")
    set(GMP_URL "https://github.com/CGAL/cgal/releases/download/v5.5/CGAL-5.5-win64-auxiliary-libraries-gmp-mpfr.zip")
    set(CGAL_URL "https://github.com/CGAL/cgal/releases/download/v5.5/CGAL-5.5-library.tar.xz" )
    FetchContent_Declare(GMP URL ${GMP_URL})
    FetchContent_MakeAvailable(GMP)
    target_link_libraries(geometry INTERFACE GMP)

    set(CGAL_ARCHIVE_NAME "CGAL-5.5-library.tar.xz")
    set(CGAL_ARCHIVE ${CURRENT_EXTERNAL_DIR}/${CGAL_ARCHIVE_NAME})
    set(CGAL_ROOT_DIR ${CURRENT_EXTERNAL_DIR}/CGAL)
    file(DOWNLOAD ${CGAL_URL} ${CGAL_ARCHIVE})
    add_custom_target(unpack_cgal ALL)
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E remove_directory "CGAL"
      WORKING_DIRECTORY ${CURRENT_EXTERNAL_DIR})
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E tar xf ${CGAL_ARCHIVE_NAME}
      WORKING_DIRECTORY ${CURRENT_EXTERNAL_DIR})
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E rename "CGAL-5.5" "CGAL"
      WORKING_DIRECTORY ${CURRENT_EXTERNAL_DIR})
    target_include_directories(geometry INTERFACE ${CGAL_ROOT_DIR}/include)
    #add_subdirectory(${CGAL_ROOT_DIR})
    set(TATOOINE_CGAL_AVAILABLE 1)
    set(TATOOINE_USE_INSTALLED_CGAL 0)
  endif()
endif()
set(TATOOINE_USE_INSTALLED_CGAL ${TATOOINE_USE_INSTALLED_CGAL} PARENT_SCOPE)
set(TATOOINE_CGAL_AVAILABLE ${TATOOINE_CGAL_AVAILABLE} PARENT_SCOPE)
target_compile_definitions(geometry PUBLIC TATOOINE_CGAL_AVAILABLE=${TATOOINE_CGAL_AVAILABLE})


if (TATOOINE_BUILD_EXAMPLES)
  add_subdirectory(examples EXCLUDE_FROM_ALL)
endif()
