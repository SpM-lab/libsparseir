# Julia

## Set up

1. Install Julia
1. Build this project:

   ```sh
   julia -e 'using Pkg; Pkg.build()'
   ```
1. It will create `src/C_API.jl` that wraps C-API of libsparseir for Julia interface.
1. To test our Julia project, run:
   ```sh
   julia -e 'using Pkg; Pkg.test()'
   ```
We use [ReTestItems.jl](https://github.com/JuliaTesting/ReTestItems.jl) as a test framework.
