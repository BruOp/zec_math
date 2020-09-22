# zec_math

Just a crappy scalar math lib that I'm using across a few different projects.

Inspired by Sean Barrett's `stb` libraries:
https://github.com/nothings/stb


## Usage

Do this:
```cpp
#define ZEC_MATH_IMPLEMENTATION
```
before you include this file in *one* C or C++ file to create the implementation.

It should look like this:
```cpp
#include ...
#define ZEC_MATH_IMPLEMENTATION
#include "zec_math.h"
```

## Notes:

- Vectors are _column-major_ and we use _post-multiplication_
so to transform a vector you would do M * v
- Matrices are layed out in row-major form, so data[1][2] refers to the third element
in the second row (A_2,3)
- To use the matrices produced by this lib with HLSL youll have to enable row-major storage.
