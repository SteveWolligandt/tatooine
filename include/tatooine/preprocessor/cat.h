#ifndef TATOOINE_PREPROCESSOR_CAT_H
#define TATOOINE_PREPROCESSOR_CAT_H
//==============================================================================
#define TATOOINE_PP_CAT(a, b) TATOOINE_PP_CAT_I(a, b)
#define TATOOINE_PP_CAT_I(a, b) a##b
//==============================================================================
#endif
