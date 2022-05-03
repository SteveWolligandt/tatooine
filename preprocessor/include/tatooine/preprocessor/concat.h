#ifndef TATOOINE_PREPROCESSOR_CONCAT_H
#define TATOOINE_PREPROCESSOR_CONCAT_H
//==============================================================================
#define TATOOINE_PP_CONCAT_(a, b) a##b
#define TATOOINE_PP_CONCAT(a, b) TATOOINE_PP_CONCAT_(a, b)
//==============================================================================
#endif
