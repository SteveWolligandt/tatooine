# Predefined Analytical Fields
#### Double Gyre
``` c++
#include <tatooine/analytical/fields/numerical/doublegyre.h>
void use_doublegyre() {
  tatooine::analytical::fields::numerical::doublegyre v;
  auto sample = v({0.1, 0.2}, 3.0);
}
```

# Operations on fields
For simple operations the pipe operator can be used:
``` c++
#include <tatooine/analytical/fields/numerical/doublegyre.h>
void apply_some_operation() {
  tatooine::analytical::fields::numerical::doublegyre v;
  auto v_double_length = 
    v | [](auto const& sample) { return length(sample); }
      | [](auto const& length) { return length * 2; };

  v_double_length({0.1, 0.2}, 3.0);
}
```

# Algorithms
#### Marching Cubes
#### Parallel Vectors
#### Stream Surface
