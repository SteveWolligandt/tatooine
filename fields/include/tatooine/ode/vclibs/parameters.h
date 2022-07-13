#ifndef TATOOINE_ODE_VCLIBS_PARAMETERS_H
#define TATOOINE_ODE_VCLIBS_PARAMETERS_H
//==============================================================================
#include <vcode/odeint.hh>
#include <tatooine/concepts.h>
//==============================================================================
namespace tatooine::ode::vclibs {
//==============================================================================
static constexpr inline auto rk43          = VC::odeint::RK43;
static constexpr inline auto out_of_domain = VC::odeint::OutOfDomain;
static constexpr inline auto stopped       = VC::odeint::evstate_t::Stopped;
static constexpr inline auto failed        = VC::odeint::evstate_t::Failed;
static constexpr inline auto ok            = VC::odeint::evstate_t::OK;
// options
static constexpr inline auto abs_tol       = VC::odeint::AbsTol;
static constexpr inline auto rel_tol       = VC::odeint::RelTol;
static constexpr inline auto initial_step  = VC::odeint::InitialStep;
static constexpr inline auto max_step      = VC::odeint::MaxStep;
static constexpr inline auto max_num_steps = VC::odeint::MaxNumSteps;
//==============================================================================
template <typename T>
struct is_option_impl : std::false_type{};
//==============================================================================
template <typename klass, typename T>
struct is_option_impl<VC::odeint::options::option_t<klass, T>>
    : std::true_type {};
//==============================================================================
template <typename T>
static auto constexpr is_option = is_option_impl<T>::value;
//==============================================================================
template <typename T>
concept option = is_option<T>;
//==============================================================================
}  // namespace tatooine::ode::vclibs
//==============================================================================
#endif
