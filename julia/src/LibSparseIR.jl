module LibSparseIR

include("C_API.jl") # LibSparseIR_C_API
using .LibSparseIR_C_API

greet() = print("Hello World!")

end # module LibSparseIR
