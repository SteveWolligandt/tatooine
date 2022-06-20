#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
add_library(tensor INTERFACE)
target_compile_features(tensor INTERFACE cxx_std_20)
target_include_directories(tensor INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
update_git_submodule(${CMAKE_CURRENT_SOURCE_DIR}/external/blaspp)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/blaspp)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
update_git_submodule(${CMAKE_CURRENT_SOURCE_DIR}/external/lapackpp)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/lapackpp)
target_compile_options(lapackpp INTERFACE -Wno-unused-parameter)
target_compile_definitions(tensor INTERFACE LAPACK_COMPLEX_CPP)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
target_link_libraries(tensor
 INTERFACE
  base
  multidim_array
  blaspp
  lapackpp)

#-------------------------------------------------------------------------------
# Get blaspp properties for tatooineConfig.cmake
# include directories
get_target_property(BLASPP_INCLUDE_DIRECTORIES blaspp INCLUDE_DIRECTORIES)
string(REPLACE "$" ";" BLASPP_INCLUDE_DIRECTORIES ${BLASPP_INCLUDE_DIRECTORIES})
foreach(INCLUDE_DIRECTORY ${BLASPP_INCLUDE_DIRECTORIES})
  set(BUILD_INTERFACE_REGEX "\<BUILD_INTERFACE:(.*)\>")
  if(INCLUDE_DIRECTORY MATCHES ${BUILD_INTERFACE_REGEX})
    string(REGEX REPLACE "(\<BUILD_INTERFACE:)" "" INCLUDE_DIRECTORY ${INCLUDE_DIRECTORY})
    string(LENGTH ${INCLUDE_DIRECTORY} len)
    math(EXPR len ${len}-1)
    string(SUBSTRING ${INCLUDE_DIRECTORY} 0 ${len} INCLUDE_DIRECTORY)
    list(APPEND FILTERED_BLASPP_INCLUDE_DIRECTORIES ${INCLUDE_DIRECTORY})
  elseif(NOT (INCLUDE_DIRECTORY MATCHES "\<.*_INTERFACE:.*\>"))
    list(APPEND FILTERED_BLASPP_INCLUDE_DIRECTORIES ${INCLUDE_DIRECTORY})
  endif()
endforeach()
set(BLASPP_INCLUDE_DIRECTORIES ${FILTERED_BLASPP_INCLUDE_DIRECTORIES} PARENT_SCOPE)

# link libraries
get_target_property(BLASPP_LINK_LIBRARIES blaspp LINK_LIBRARIES)
message(STATUS "${BLASPP_LINK_LIBRARIES}")
set(BLASPP_LINK_LIBRARIES ${BLASPP_LINK_LIBRARIES} PARENT_SCOPE)

# compile features
get_target_property(BLASPP_COMPILE_FEATURES blaspp COMPILE_FEATURES)
set(BLASPP_COMPILE_FEATURES ${BLASPP_COMPILE_FEATURES} PARENT_SCOPE)

# compile options
get_target_property(BLASPP_COMPILE_OPTIONS blaspp COMPILE_OPTIONS)
message(STATUS "${BLASPP_COMPILE_OPTIONS}")
#-------------------------------------------------------------------------------
get_target_property(LAPACKPP_INCLUDE_DIRECTORIES lapackpp INCLUDE_DIRECTORIES)
string(REPLACE "$" ";" LAPACKPP_INCLUDE_DIRECTORIES ${LAPACKPP_INCLUDE_DIRECTORIES})
foreach(INCLUDE_DIRECTORY ${LAPACKPP_INCLUDE_DIRECTORIES})
  set(BUILD_INTERFACE_REGEX "\<BUILD_INTERFACE:(.*)\>")
  if(INCLUDE_DIRECTORY MATCHES ${BUILD_INTERFACE_REGEX})
    string(REGEX REPLACE "(\<BUILD_INTERFACE:)" "" INCLUDE_DIRECTORY ${INCLUDE_DIRECTORY})
    string(LENGTH ${INCLUDE_DIRECTORY} len)
    math(EXPR len ${len}-1)
    string(SUBSTRING ${INCLUDE_DIRECTORY} 0 ${len} INCLUDE_DIRECTORY)
    list(APPEND FILTERED_LAPACKPP_INCLUDE_DIRECTORIES ${INCLUDE_DIRECTORY})
  elseif(NOT (INCLUDE_DIRECTORY MATCHES "\<.*_INTERFACE:.*\>"))
    list(APPEND FILTERED_LAPACKPP_INCLUDE_DIRECTORIES ${INCLUDE_DIRECTORY})
  endif()
endforeach()
set(LAPACKPP_INCLUDE_DIRECTORIES ${FILTERED_LAPACKPP_INCLUDE_DIRECTORIES} PARENT_SCOPE)

# link libraries
get_target_property(LAPACKPP_LINK_LIBRARIES lapackpp LINK_LIBRARIES)
set(LAPACKPP_LINK_LIBRARIES ${LAPACKPP_LINK_LIBRARIES} PARENT_SCOPE)

# compile features
get_target_property(LAPACKPP_COMPILE_FEATURES lapackpp COMPILE_FEATURES)
set(LAPACKPP_COMPILE_FEATURES ${LAPACKPP_COMPILE_FEATURES} PARENT_SCOPE)

# compile options
get_target_property(LAPACKPP_COMPILE_OPTIONS lapackpp COMPILE_OPTIONS)
message(STATUS "${LAPACKPP_COMPILE_OPTIONS}")
