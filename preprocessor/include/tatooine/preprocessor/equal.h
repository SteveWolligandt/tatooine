#ifndef TATOOINE_PRERPOCESSOR_EQUAL_H
#define TATOOINE_PRERPOCESSOR_EQUAL_H
//==============================================================================
#include <tatooine/preprocessor/not_equal.h>
#include <tatooine/preprocessor/compl.h>
//==============================================================================
#define TATOOINE_PP_EQUAL(x, y) TATOOINE_PP_COMPL(TATOOINE_PP_NOT_EQUAL(x, y))
//==============================================================================
#endif
