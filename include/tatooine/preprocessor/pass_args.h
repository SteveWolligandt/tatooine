#include <tatooine/preprocessor/strip_parantheses.h>
#include <tatooine/preprocessor/va_args.h>
#ifndef TATOOINE_PP_PASS_ARGS
#define TATOOINE_PP_PASS_ARGS(X) TATOOINE_ESC(TATOOINE_ISH X)
#define TATOOINE_ISH(...) TATOOINE_ISH __VA_ARGS__
#define TATOOINE_ESC(...) TATOOINE_ESC_(__VA_ARGS__)
#define TATOOINE_ESC_(...) TATOOINE_VAN ## __VA_ARGS__
#define TATOOINE_VANTATOOINE_ISH
#endif
