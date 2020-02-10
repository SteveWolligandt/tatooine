find_package(Doxygen)
find_package(Sphinx)
if (SPHINX_FOUND)
  option(TATOOINE_DOC_USE_SPHINX "Use Sphinx to build documentation" FALSE)
else()
  set(TATOOINE_DOC_USE_SPHINX "Use Sphinx to build documentation" FALSE)
endif()
if (SPHINX_FOUND AND TATOOINE_DOC_USE_SPHINX)
  set(TATOOINE_DOC_DOXYGEN_MESSAGE "Building XML documentation with Doxygen")
  set(TATOOINE_DOC_DOXYGEN_GEN_HTML NO)
  set(TATOOINE_DOC_DOXYGEN_GEN_XML  YES)
else()
  set(TATOOINE_DOC_DOXYGEN_MESSAGE "Building HTML documentation with Doxygen")
  set(TATOOINE_DOC_DOXYGEN_GEN_HTML YES)
  set(TATOOINE_DOC_DOXYGEN_GEN_XML  NO)
endif()
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
if (DOXYGEN_FOUND)
  set(TATOOINE_DOC_SRC_DOXYFILE      ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in )
  set(TATOOINE_DOC_BUILD_DOXYFILE    ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile    )
  set(TATOOINE_DOC_DOXYGEN_HTML_DIR  ${CMAKE_CURRENT_BINARY_DIR}/html)
  configure_file(${TATOOINE_DOC_SRC_DOXYFILE} ${TATOOINE_DOC_BUILD_DOXYFILE})

  add_custom_target(doc.doxygen ALL
    COMMAND doxygen ${TATOOINE_DOC_BUILD_DOXYFILE}
    COMMENT ${TATOOINE_DOC_DOXYGEN_MESSAGE}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  if (SPHINX_FOUND AND TATOOINE_DOC_USE_SPHINX)
    set(TATOOINE_DOC_SPHINX_THEME sphinx_rtd_theme)
    if(NOT DEFINED TATOOINE_DOC_SPHINX_THEME)
      set(TATOOINE_DOC_SPHINX_THEME default)
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
