set(DOC_IMAGE_DIR ${CMAKE_CURRENT_BINARY_DIR}/images)

#find_package(Sphinx REQUIRED)

#set(SPHINX_THEME sphinx_rtd_theme)
#if(NOT DEFINED SPHINX_THEME)
#  set(SPHINX_THEME default)
#endif()
#
#if(NOT DEFINED SPHINX_THEME_DIR)
#  set(SPHINX_THEME_DIR)
#endif()
#
#set(BINARY_BUILD_DIR  ${CMAKE_CURRENT_BINARY_DIR}/_build      )
#set(SPHINX_CACHE_DIR  ${CMAKE_CURRENT_BINARY_DIR}/_doctrees   )
#set(HTML_DIR          ${CMAKE_CURRENT_BINARY_DIR}/html        )
#set(SRC_SPHINX_CONF   ${CMAKE_CURRENT_SOURCE_DIR}/conf.py.in  )
#set(BUILD_SPHINX_CONF ${CMAKE_CURRENT_BINARY_DIR}/conf.py     )
#configure_file(${SRC_SPHINX_CONF}                        ${BUILD_SPHINX_CONF})
#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/index.rst     ${CMAKE_CURRENT_BINARY_DIR}/index.rst)
#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/building.rst  ${CMAKE_CURRENT_BINARY_DIR}/building.rst)
#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/todo_list.rst ${CMAKE_CURRENT_BINARY_DIR}/todo_list.rst)

set(SRC_DOXYFILE           ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in )
set(BUILD_DOXYFILE         ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile    )
set(DOXYGEN_GENERATE_LATEX NO)
set(DOXYGEN_GENERATE_XML   NO)
set(DOXYGEN_GENERATE_HTML  YES)
set(DOXYGEN_HTML_DIR       ${CMAKE_CURRENT_BINARY_DIR}/html_doxygen)

configure_file(${SRC_DOXYFILE}                           ${BUILD_DOXYFILE})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/images DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_custom_target(doc.doxygen ALL
  COMMAND rm -rf ${DOXYGEN_HTML_DIR}
  COMMAND doxygen ${BUILD_DOXYFILE}
  COMMENT "Building HTML documentation with Doxygen"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

#add_custom_target(doc.sphinx ALL
#  COMMAND mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/_static
#  COMMAND ${SPHINX_EXECUTABLE}
#    -q -b html
#    -d ${SPHINX_CACHE_DIR}
#    ${CMAKE_CURRENT_BINARY_DIR}
#    ${HTML_DIR}
#  COMMENT "Building HTML documentation with Sphinx"
#  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#  DEPENDS doc.doxygen convert_result_to_jpg
#)

add_custom_target(doc ALL
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS doc.doxygen
)
