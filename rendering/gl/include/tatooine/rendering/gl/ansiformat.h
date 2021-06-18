#ifndef YAVIN_ANSI_FORMAT_H
#define YAVIN_ANSI_FORMAT_H
//==============================================================================
#include <string>
//==============================================================================
namespace yavin::ansi {
//==============================================================================
static constexpr std::string_view black   = "\033[30m";
static constexpr std::string_view red     = "\033[31m";
static constexpr std::string_view green   = "\033[32m";
static constexpr std::string_view yellow  = "\033[33m";
static constexpr std::string_view blue    = "\033[34m";
static constexpr std::string_view magenta = "\033[35m";
static constexpr std::string_view cyan    = "\033[36m";
static constexpr std::string_view white   = "\033[37m";

static constexpr std::string_view black_bg   = "\033[40m";
static constexpr std::string_view red_bg     = "\033[41m";
static constexpr std::string_view green_bg   = "\033[42m";
static constexpr std::string_view yellow_bg  = "\033[43m";
static constexpr std::string_view blue_bg    = "\033[44m";
static constexpr std::string_view magenta_bg = "\033[45m";
static constexpr std::string_view cyan_bg    = "\033[46m";
static constexpr std::string_view white_bg   = "\033[47m";

static constexpr std::string_view reset         = "\033[0m";
static constexpr std::string_view bold          = "\033[1m";
static constexpr std::string_view underline     = "\033[4m";
static constexpr std::string_view inverse       = "\033[7m";
static constexpr std::string_view bold_off      = "\033[21m";
static constexpr std::string_view underline_off = "\033[24m";
static constexpr std::string_view inverse_off   = "\033[27m";
//==============================================================================
}  // namespace yavin::ansi
//==============================================================================
#endif
