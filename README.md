[Doxygen Documentation](https://pages.vc.cs.ovgu.de/tatooine/index.html)

# Grid
#### Basic usage
``` c++
#include <tatooine/rectilinear_grid.h>
void use_linearly_spaced_rectilinear_grid() {
  tatooine::rectilinear_grid gr{
      64, 64, 64};  // This creates a three-dimensional rectilinear grid with
                    // a resolution of 64x64x64. It ranges in [0,1]x[0,1]x[0,1]
}
```

# Predefined Analytical Fields
#### Double Gyre
``` c++
#include <tatooine/analytical/numerical/doublegyre.h>
void use_doublegyre() {
  tatooine::analytical::numerical::doublegyre v;
  auto sample = v({0.1, 0.2}, 3.0);
}
```

# Operations on fields
For simple operations the pipe operator can be used:
``` c++
#include <tatooine/analytical/numerical/doublegyre.h>
void apply_some_operation() {
  tatooine::analytical::numerical::doublegyre v;
  auto v_double_length = 
    v | [](auto const& sample) { return length(sample); }
      | [](auto const& length) { return length * 2; };

  v_double_length({0.1, 0.2}, 3.0);
}
```
