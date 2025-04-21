using Libdl: dlext
# libsparseir = expanduser("~/opt/libsparseir/lib/libsparseir.$(dlext)")
libsparseir = expanduser(joinpath(@__DIR__, "../build/libsparseir.$(dlext)"))

mutable struct _spir_bosonic_finite_temp_basis end

const spir_bosonic_finite_temp_basis = _spir_bosonic_finite_temp_basis

function spir_bosonic_finite_temp_basis_new(beta, omega_max, epsilon)
    ccall((:spir_bosonic_finite_temp_basis_new, libsparseir), Ptr{spir_bosonic_finite_temp_basis}, (Cdouble, Cdouble, Cdouble), beta, omega_max, epsilon)
end

b = spir_bosonic_finite_temp_basis_new(0.1, 0.3, 0.0001)
