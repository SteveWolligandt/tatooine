find_package(Doxygen)
find_package(Sphinx QUIET)
if (SPHINX_FOUND)
  option(TATOOINE_DOC_USE_SPHINX "Use Sphinx to build documentation" TRUE)
else()
  set(TATOOINE_DOC_USE_SPHINX "Use Sphinx to build documentation" FALSE)
endif()
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
if (DOXYGEN_FOUND)
  set(TATOOINE_DOC_HTML_DIR             ${CMAKE_CURRENT_BINARY_DIR}/html      )
  set(TATOOINE_DOC_SOURCE_DOXYFILE      ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.base.in)
  set(TATOOINE_DOC_BINARY_DOXYFILE      ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.base   )
  set(TATOOINE_DOC_SOURCE_HTML_DOXYFILE ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.html.in)
  set(TATOOINE_DOC_BINARY_HTML_DOXYFILE ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.html   )
  set(TATOOINE_DOC_SOURCE_XML_DOXYFILE  ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.xml.in )
  set(TATOOINE_DOC_BINARY_XML_DOXYFILE  ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.xml    )
  set(TATOOINE_DOC_SOURCE_PDF_DOXYFILE  ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.pdf.in )
  set(TATOOINE_DOC_BINARY_PDF_DOXYFILE  ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.pdf    )
  set(TATOOINE_DOC_DOXYGEN_HTML_MESSAGE "Building HTML documentation with Doxygen")
  set(TATOOINE_DOC_DOXYGEN_XML_MESSAGE  "Building XML documentation with Doxygen")
  set(TATOOINE_DOC_DOXYGEN_PDF_MESSAGE  "Building PDF documentation with Doxygen")
  configure_file(${TATOOINE_DOC_SOURCE_DOXYFILE} ${TATOOINE_DOC_BINARY_DOXYFILE})
  configure_file(${TATOOINE_DOC_SOURCE_HTML_DOXYFILE} ${TATOOINE_DOC_BINARY_HTML_DOXYFILE})
  configure_file(${TATOOINE_DOC_SOURCE_XML_DOXYFILE} ${TATOOINE_DOC_BINARY_XML_DOXYFILE})
  configure_file(${TATOOINE_DOC_SOURCE_PDF_DOXYFILE} ${TATOOINE_DOC_BINARY_PDF_DOXYFILE})

  add_custom_target(doc.doxygen.html ALL
    COMMAND doxygen ${TATOOINE_DOC_BINARY_HTML_DOXYFILE}
    COMMENT ${TATOOINE_DOC_DOXYGEN_HTML_MESSAGE})
  add_custom_target(doc.doxygen.xml ALL
    COMMAND doxygen ${TATOOINE_DOC_BINARY_XML_DOXYFILE}
    COMMENT ${TATOOINE_DOC_DOXYGEN_XML_MESSAGE})
  add_custom_target(doc.doxygen.pdf ALL
    COMMAND doxygen ${TATOOINE_DOC_BINARY_PDF_DOXYFILE}
    COMMAND make -C${CMAKE_CURRENT_BINARY_DIR}/latex pdf
    COMMENT ${TATOOINE_DOC_DOXYGEN_PDF_MESSAGE})

  if (SPHINX_FOUND AND TATOOINE_DOC_USE_SPHINX)
    set(TATOOINE_DOC_SPHINX_THEME sphinx_rtd_theme)
    if(NOT DEFINED TATOOINE_DOC_SPHINX_THEME)
      set(TATOOINE_DOC_SPHINX_THEME default)
    endif()

    set(BINARY_BINARY_DIR  ${CMAKE_CURRENT_BINARY_DIR}/_build    )
    set(SPHINX_CACHE_DIR  ${CMAKE_CURRENT_BINARY_DIR}/_doctrees )
    set(SOURCE_SPHINX_CONF   ${CMAKE_CURRENT_SOURCE_DIR}/conf.py.in)
    set(BINARY_SPHINX_CONF ${CMAKE_CURRENT_BINARY_DIR}/conf.py   )

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/index.rst
                   ${CMAKE_CURRENT_BINARY_DIR}/index.rst)
    configure_file(${SOURCE_SPHINX_CONF}
                   ${BINARY_SPHINX_CONF})

    add_custom_target(doc.sphinx ALL
      COMMAND mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/_static
      COMMAND ${SPHINX_EXECUTABLE}
        -q -b html
        -d ${SPHINX_CACHE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
        ${TATOOINE_DOC_HTML_DIR}
      COMMENT "Building HTML documentation with Sphinx"
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      DEPENDS doc.doxygen.xml)

    add_custom_target(doc ALL
      COMMENT "Building HTML documentation with Sphinx"
      DEPENDS doc.sphinx)
  else()
    add_custom_target(doc ALL
      COMMENT ${TATOOINE_DOC_BINARY_HTML_DOXYFILE}
      DEPENDS doc.doxygen.html)
  endif()
else()
  message(STATUS "Will not be able to generate documentation!")
endif()
