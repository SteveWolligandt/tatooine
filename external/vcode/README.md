ODE
===

Solve *ordinary differential equations* (ODE) that typically arise in flow visiualization:

- Small dimensions (use `VC::vecn::vecn`).
- Sufficiently smooth functions: focus on Runge-Kutta methods.
- Must be able to handle boundaries conveniently.
- Dense output *and* refine-able output.

Example: [example_ode.cc](example_ode.cc)
