# - Find NetCDF
# Find the native NetCDF includes and library
#
#  NETCDF_INCLUDES    - where to find netcdf.h, etc
#  NETCDF_LIBRARIES   - Link these libraries when using NetCDF
#  NETCDF_FOUND       - True if NetCDF found including required interfaces (see below)
#
# The following are not for general use and are included in
# NETCDF_LIBRARIES if the corresponding option above is set.
#
#  NETCDF_LIBRARIES_C           - Just the C interface
#  NETCDF_LIBRARIES_CXX         - C++ interface, if available
#
# Normal usage would be:
#  find_package (NetCDF REQUIRED)
#  target_link_libraries (uses_netcdf ${NETCDF_LIBRARIES})

if (NETCDF_INCLUDES AND NETCDF_LIBRARIES)
  # Already in cache, be silent
  set (NETCDF_FIND_QUIETLY TRUE)
endif (NETCDF_INCLUDES AND NETCDF_LIBRARIES)

find_path (NETCDF_INCLUDES netcdf.h
  HINTS NETCDF_DIR ENV NETCDF_DIR)

find_library (NETCDF_LIBRARIES_C       NAMES netcdf)
mark_as_advanced(NETCDF_LIBRARIES_C)

set (NetCDF_has_interfaces "YES") # will be set to NO if we're missing any interfaces
set (NetCDF_libs ${NETCDF_LIBRARIES_C})

get_filename_component (NetCDF_lib_dirs "${NETCDF_LIBRARIES_C}" PATH)

macro(NetCDF_check_interface lang header libs)
  message(STATUS "got NETCDF_${lang}")
  find_path (NETCDF_INCLUDES_${lang} NAMES ${header}
    HINTS "${NETCDF_INCLUDES}" NO_DEFAULT_PATH)
  find_library (NETCDF_LIBRARIES_${lang} NAMES ${libs}
    HINTS "${NetCDF_lib_dirs}" NO_DEFAULT_PATH)
  mark_as_advanced (NETCDF_INCLUDES_${lang} NETCDF_LIBRARIES_${lang})
  if (NETCDF_INCLUDES_${lang} AND NETCDF_LIBRARIES_${lang})
    list (APPEND NetCDF_libs  ${NETCDF_LIBRARIES_${lang}}) # prepend so that -lnetcdf is last
  else (NETCDF_INCLUDES_${lang} AND NETCDF_LIBRARIES_${lang})
    set (NetCDF_has_interfaces "NO")
    message (STATUS "Failed to find NetCDF interface for ${lang}")
  endif (NETCDF_INCLUDES_${lang} AND NETCDF_LIBRARIES_${lang})
endmacro()

NetCDF_check_interface (CXX netcdf netcdf_c++4)

set (NETCDF_LIBRARIES "${NetCDF_libs}" CACHE STRING "All NetCDF libraries required for interface level")

# handle the QUIETLY and REQUIRED arguments and set NETCDF_FOUND to TRUE if
# all listed variables are TRUE
include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (NetCDF DEFAULT_MSG NETCDF_LIBRARIES NETCDF_INCLUDES NetCDF_has_interfaces)

mark_as_advanced (NETCDF_LIBRARIES NETCDF_INCLUDES)
