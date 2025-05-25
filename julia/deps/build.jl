using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Clang.Generators
using Clang.LibClang.Clang_jll

include_dir = normpath(joinpath(@__DIR__, "../../include"))
sparseir_dir = joinpath(include_dir, "sparseir")

# wrapper generator options
options = load_options(joinpath(@__DIR__, "generator.toml"))

# add compiler flags, e.g. "-DXXXXXXXXX"
args = get_default_args()
push!(args, "-I$include_dir")

headers = [joinpath(sparseir_dir, header) for header in readdir(sparseir_dir) if endswith(header, ".h")]
# there is also an experimental `detect_headers` function for auto-detecting top-level headers in the directory
# headers = detect_headers(sparseir_dir, args)

# create context
ctx = create_context(headers, args, options)

# run generator
build!(ctx)

run(`sed -i '' 's/const c_complex = ComplexF32/const c_complex = ComplexF64/g' $(joinpath(@__DIR__, "../src/C_API.jl"))`)
