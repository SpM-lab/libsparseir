# Julia

## Set up

1. Install Julia
1. Install Clang.jl via:

    ```sh
    julia -e 'using Pkg; Pkg.activate(); Pkg.add("Clang")'
    ```

    Clang.jl parses header files in the form of "../incluse/sparseir/*.h"
1. Run `julia build.jl` to generate Julia wrapper for `libsparseir`. It will create `C_API.jl` that wraps C-API of libsparseir for Julia interface.
