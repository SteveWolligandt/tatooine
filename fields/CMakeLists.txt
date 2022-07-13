add_library(fields INTERFACE)
target_include_directories(fields
 INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  #${CMAKE_CURRENT_BINARY_DIR}/include
  )
target_link_libraries(fields
 INTERFACE
  tensor
  geometry)
#-------------------------------------------------------------------------------
# Get fields properties for tatooineConfig.cmake
# include directories
get_target_property(FIELDS_INCLUDE_DIRECTORIES fields INTERFACE_INCLUDE_DIRECTORIES)
foreach(INCLUDE_DIRECTORY ${FIELDS_INCLUDE_DIRECTORIES})
  if(${INCLUDE_DIRECTORY} MATCHES ".*BUILD_INTERFACE:.*")
    string(LENGTH ${INCLUDE_DIRECTORY} len)
    math(EXPR len ${len}-19)
    string(SUBSTRING ${INCLUDE_DIRECTORY} 18 ${len} INCLUDE_DIRECTORY)
    list(APPEND FILTERED_FIELDS_INCLUDE_DIRECTORIES ${INCLUDE_DIRECTORY})
  elseif(NOT (INCLUDE_DIRECTORY MATCHES "\<.*_INTERFACE:.*\>"))
    list(APPEND FILTERED_FIELDS_INCLUDE_DIRECTORIES ${INCLUDE_DIRECTORY})
  else()
  endif()
endforeach()
set(FIELDS_INCLUDE_DIRECTORIES ${FILTERED_FIELDS_INCLUDE_DIRECTORIES} PARENT_SCOPE)

get_target_property(FIELDS_LINK_LIBRARIES fields INTERFACE_LINK_LIBRARIES)
foreach(LINK_LIBRARY ${FIELDS_LINK_LIBRARIES})
  if(${LINK_LIBRARY} MATCHES "tensor")
    list(APPEND FILTERED_FIELDS_LINK_LIBRARIES "tatooine::tensor")
  elseif(${LINK_LIBRARY} MATCHES "geometry")
    list(APPEND FILTERED_FIELDS_LINK_LIBRARIES "tatooine::geometry")
  elseif(NOT (LINK_LIBRARY MATCHES "\<.*_INTERFACE:.*\>"))
    list(APPEND FILTERED_FIELDS_LINK_LIBRARIES ${LINK_LIBRARY})
  endif()
endforeach()
set(FIELDS_LINK_LIBRARIES ${FILTERED_FIELDS_LINK_LIBRARIES} PARENT_SCOPE)
