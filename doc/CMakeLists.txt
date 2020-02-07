find_package(Doxygen)
find_package(Sphinx)
if (SPHINX_FOUND)
  message(STATUS "Sphinx was found!")
  option(USE_SPHINX "Use Sphinx to build documentation" FALSE)
else()
  message(STATUS "Sphinx not found")
  set(USE_SPHINX FALSE)
endif()
if (SPHINX_FOUND AND USE_SPHINX)
  set(DOXYGEN_GEN_HTML NO)
  set(DOXYGEN_GEN_XML  YES)
else()
  set(DOXYGEN_GEN_HTML YES)
  set(DOXYGEN_GEN_XML  NO)
endif()
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
if (DOXYGEN_FOUND)
  set(SRC_DOXYFILE      ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in )
  set(BUILD_DOXYFILE    ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile    )
  #set(DOC_IMAGE_DIR     ${CMAKE_CURRENT_BINARY_DIR}/images      )
  set(DOXYGEN_HTML_DIR  ${CMAKE_CURRENT_BINARY_DIR}/html)

  configure_file(${SRC_DOXYFILE} ${BUILD_DOXYFILE})
  #file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/images
  #     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

  add_custom_target(doc.doxygen ALL
    # COMMAND rm -rf ${DOXYGEN_HTML_DIR}
    COMMAND doxygen ${BUILD_DOXYFILE}
    COMMENT "Building HTML documentation with Doxygen"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  if (SPHINX_FOUND AND USE_SPHINX)
    set(SPHINX_THEME sphinx_rtd_theme)
    if(NOT DEFINED SPHINX_THEME)
      set(SPHINX_THEME default)
    endif()

    if(NOT DEFINED SPHINX_THEME_DIR)
      set(SPHINX_THEME_DIR)
    endif()

    set(HTML_DIR          ${CMAKE_CURRENT_BINARY_DIR}/html      )
    set(BINARY_BUILD_DIR  ${CMAKE_CURRENT_BINARY_DIR}/_build    )
    set(SPHINX_CACHE_DIR  ${CMAKE_CURRENT_BINARY_DIR}/_doctrees )
    set(SRC_SPHINX_CONF   ${CMAKE_CURRENT_SOURCE_DIR}/conf.py.in)
    set(BUILD_SPHINX_CONF ${CMAKE_CURRENT_BINARY_DIR}/conf.py   )

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/index.rst
                   ${CMAKE_CURRENT_BINARY_DIR}/index.rst)
    configure_file(${SRC_SPHINX_CONF}
                   ${BUILD_SPHINX_CONF})

    add_custom_target(doc.sphinx ALL
      COMMAND mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/_static
      COMMAND ${SPHINX_EXECUTABLE}
        -q -b html
        -d ${SPHINX_CACHE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
        ${HTML_DIR}
      COMMENT "Building HTML documentation with Sphinx"
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      DEPENDS doc.doxygen)

    add_custom_target(doc ALL
      COMMENT "Building HTML documentation with Sphinx"
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      DEPENDS doc.doxygen doc.sphinx)
  else()
    add_custom_target(doc ALL
      COMMENT "Building HTML documentation with Sphinx"
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      DEPENDS doc.doxygen)
  endif()
else()
  message(STATUS "Will not be able to generate documentation!")
endif()
