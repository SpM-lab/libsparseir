module LibSparseIR_C_API

using CEnum

using Libdl: dlext

libsparseir = expanduser("~/opt/libsparseir/lib/libsparseir.$(dlext)")


const c_complex = ComplexF32

@cenum int32_t::UInt32 begin
    SPIR_STATISTICS_FERMIONIC = 1
    SPIR_STATISTICS_BOSONIC = 0
end

@cenum int32_t::UInt32 begin
    SPIR_ORDER_COLUMN_MAJOR = 1
    SPIR_ORDER_ROW_MAJOR = 0
end

mutable struct _spir_kernel end

const spir_kernel = _spir_kernel

function spir_release_kernel(obj)
    ccall((:spir_release_kernel, libsparseir), Cvoid, (Ptr{spir_kernel},), obj)
end

function spir_clone_kernel(src)
    ccall((:spir_clone_kernel, libsparseir), Ptr{spir_kernel}, (Ptr{spir_kernel},), src)
end

function spir_is_assigned_kernel(obj)
    ccall((:spir_is_assigned_kernel, libsparseir), Cint, (Ptr{spir_kernel},), obj)
end

mutable struct _spir_logistic_kernel end

const spir_logistic_kernel = _spir_logistic_kernel

function spir_release_logistic_kernel(obj)
    ccall((:spir_release_logistic_kernel, libsparseir), Cvoid, (Ptr{spir_logistic_kernel},), obj)
end

function spir_clone_logistic_kernel(src)
    ccall((:spir_clone_logistic_kernel, libsparseir), Ptr{spir_logistic_kernel}, (Ptr{spir_logistic_kernel},), src)
end

function spir_is_assigned_logistic_kernel(obj)
    ccall((:spir_is_assigned_logistic_kernel, libsparseir), Cint, (Ptr{spir_logistic_kernel},), obj)
end

mutable struct _spir_reg_bose_kernel end

const spir_reg_bose_kernel = _spir_reg_bose_kernel

function spir_release_reg_bose_kernel(obj)
    ccall((:spir_release_reg_bose_kernel, libsparseir), Cvoid, (Ptr{spir_reg_bose_kernel},), obj)
end

function spir_clone_reg_bose_kernel(src)
    ccall((:spir_clone_reg_bose_kernel, libsparseir), Ptr{spir_reg_bose_kernel}, (Ptr{spir_reg_bose_kernel},), src)
end

function spir_is_assigned_reg_bose_kernel(obj)
    ccall((:spir_is_assigned_reg_bose_kernel, libsparseir), Cint, (Ptr{spir_reg_bose_kernel},), obj)
end

mutable struct _spir_continuous_functions end

const spir_continuous_functions = _spir_continuous_functions

function spir_release_continuous_functions(obj)
    ccall((:spir_release_continuous_functions, libsparseir), Cvoid, (Ptr{spir_continuous_functions},), obj)
end

function spir_clone_continuous_functions(src)
    ccall((:spir_clone_continuous_functions, libsparseir), Ptr{spir_continuous_functions}, (Ptr{spir_continuous_functions},), src)
end

function spir_is_assigned_continuous_functions(obj)
    ccall((:spir_is_assigned_continuous_functions, libsparseir), Cint, (Ptr{spir_continuous_functions},), obj)
end

mutable struct _spir_matsubara_funcs end

const spir_matsubara_funcs = _spir_matsubara_funcs

function spir_release_matsubara_funcs(obj)
    ccall((:spir_release_matsubara_funcs, libsparseir), Cvoid, (Ptr{spir_matsubara_funcs},), obj)
end

function spir_clone_matsubara_funcs(src)
    ccall((:spir_clone_matsubara_funcs, libsparseir), Ptr{spir_matsubara_funcs}, (Ptr{spir_matsubara_funcs},), src)
end

function spir_is_assigned_matsubara_funcs(obj)
    ccall((:spir_is_assigned_matsubara_funcs, libsparseir), Cint, (Ptr{spir_matsubara_funcs},), obj)
end

mutable struct _spir_fermionic_finite_temp_basis end

const spir_fermionic_finite_temp_basis = _spir_fermionic_finite_temp_basis

function spir_release_fermionic_finite_temp_basis(obj)
    ccall((:spir_release_fermionic_finite_temp_basis, libsparseir), Cvoid, (Ptr{spir_fermionic_finite_temp_basis},), obj)
end

function spir_clone_fermionic_finite_temp_basis(src)
    ccall((:spir_clone_fermionic_finite_temp_basis, libsparseir), Ptr{spir_fermionic_finite_temp_basis}, (Ptr{spir_fermionic_finite_temp_basis},), src)
end

function spir_is_assigned_fermionic_finite_temp_basis(obj)
    ccall((:spir_is_assigned_fermionic_finite_temp_basis, libsparseir), Cint, (Ptr{spir_fermionic_finite_temp_basis},), obj)
end

mutable struct _spir_bosonic_finite_temp_basis end

const spir_bosonic_finite_temp_basis = _spir_bosonic_finite_temp_basis

function spir_release_bosonic_finite_temp_basis(obj)
    ccall((:spir_release_bosonic_finite_temp_basis, libsparseir), Cvoid, (Ptr{spir_bosonic_finite_temp_basis},), obj)
end

function spir_clone_bosonic_finite_temp_basis(src)
    ccall((:spir_clone_bosonic_finite_temp_basis, libsparseir), Ptr{spir_bosonic_finite_temp_basis}, (Ptr{spir_bosonic_finite_temp_basis},), src)
end

function spir_is_assigned_bosonic_finite_temp_basis(obj)
    ccall((:spir_is_assigned_bosonic_finite_temp_basis, libsparseir), Cint, (Ptr{spir_bosonic_finite_temp_basis},), obj)
end

mutable struct _spir_sampling end

const spir_sampling = _spir_sampling

function spir_release_sampling(obj)
    ccall((:spir_release_sampling, libsparseir), Cvoid, (Ptr{spir_sampling},), obj)
end

function spir_clone_sampling(src)
    ccall((:spir_clone_sampling, libsparseir), Ptr{spir_sampling}, (Ptr{spir_sampling},), src)
end

function spir_is_assigned_sampling(obj)
    ccall((:spir_is_assigned_sampling, libsparseir), Cint, (Ptr{spir_sampling},), obj)
end

mutable struct _spir_sve_result end

const spir_sve_result = _spir_sve_result

function spir_release_sve_result(obj)
    ccall((:spir_release_sve_result, libsparseir), Cvoid, (Ptr{spir_sve_result},), obj)
end

function spir_clone_sve_result(src)
    ccall((:spir_clone_sve_result, libsparseir), Ptr{spir_sve_result}, (Ptr{spir_sve_result},), src)
end

function spir_is_assigned_sve_result(obj)
    ccall((:spir_is_assigned_sve_result, libsparseir), Cint, (Ptr{spir_sve_result},), obj)
end

mutable struct _spir_fermionic_dlr end

const spir_fermionic_dlr = _spir_fermionic_dlr

function spir_release_fermionic_dlr(obj)
    ccall((:spir_release_fermionic_dlr, libsparseir), Cvoid, (Ptr{spir_fermionic_dlr},), obj)
end

function spir_clone_fermionic_dlr(src)
    ccall((:spir_clone_fermionic_dlr, libsparseir), Ptr{spir_fermionic_dlr}, (Ptr{spir_fermionic_dlr},), src)
end

function spir_is_assigned_fermionic_dlr(obj)
    ccall((:spir_is_assigned_fermionic_dlr, libsparseir), Cint, (Ptr{spir_fermionic_dlr},), obj)
end

mutable struct _spir_bosonic_dlr end

const spir_bosonic_dlr = _spir_bosonic_dlr

function spir_release_bosonic_dlr(obj)
    ccall((:spir_release_bosonic_dlr, libsparseir), Cvoid, (Ptr{spir_bosonic_dlr},), obj)
end

function spir_clone_bosonic_dlr(src)
    ccall((:spir_clone_bosonic_dlr, libsparseir), Ptr{spir_bosonic_dlr}, (Ptr{spir_bosonic_dlr},), src)
end

function spir_is_assigned_bosonic_dlr(obj)
    ccall((:spir_is_assigned_bosonic_dlr, libsparseir), Cint, (Ptr{spir_bosonic_dlr},), obj)
end

"""
    spir_logistic_kernel_new(lambda)

Creates a new logistic kernel for fermionic/bosonic analytical continuation.

In dimensionless variables x = 2τ/β - 1, y = βω/Λ, the integral kernel is a function on [-1, 1] × [-1, 1]:

K(x, y) = exp(-Λy(x + 1)/2)/(1 + exp(-Λy))

While LogisticKernel is primarily a fermionic analytic continuation kernel, it can also model the τ dependence of a bosonic correlation function as:

∫ [exp(-Λy(x + 1)/2)/(1 - exp(-Λy))] ρ(y) dy = ∫ K(x, y) ρ'(y) dy

where ρ'(y) = w(y)ρ(y) and the weight function w(y) = 1/tanh(Λy/2)

# Arguments
* `lambda`: The cutoff parameter Λ (must be non-negative)
# Returns
A pointer to the newly created kernel object, or NULL if creation fails
"""
function spir_logistic_kernel_new(lambda)
    ccall((:spir_logistic_kernel_new, libsparseir), Ptr{spir_kernel}, (Cdouble,), lambda)
end

"""
    spir_reg_bose_kernel_new(lambda)

Creates a new regularized bosonic kernel for analytical continuation.

In dimensionless variables x = 2τ/β - 1, y = βω/Λ, the integral kernel is a function on [-1, 1] × [-1, 1]:

K(x, y) = y * exp(-Λy(x + 1)/2)/(exp(-Λy) - 1)

Special care is taken in evaluating this expression around y = 0 to handle the singularity. The kernel is specifically designed for bosonic functions and includes proper regularization to handle numerical stability issues.

!!! note

    This kernel is specifically designed for bosonic correlation functions and should not be used for fermionic cases.

# Arguments
* `lambda`: The cutoff parameter Λ (must be non-negative)
# Returns
A pointer to the newly created kernel object, or NULL if creation fails
"""
function spir_reg_bose_kernel_new(lambda)
    ccall((:spir_reg_bose_kernel_new, libsparseir), Ptr{spir_kernel}, (Cdouble,), lambda)
end

"""
    spir_sve_result_new(k, epsilon)

Perform truncated singular value expansion (SVE) of a kernel.

Computes a truncated singular value expansion of an integral kernel K: [xmin, xmax] × [ymin, ymax] → ℝ in the form:

K(x, y) = ∑ s[l] * u[l](x) * v[l](y) for l = 1, 2, 3, ...

where: - s[l] are singular values in non-increasing order - u[l](x) are left singular functions, forming an orthonormal system on [xmin, xmax] - v[l](y) are right singular functions, forming an orthonormal system on [ymin, ymax]

The SVE is computed by mapping it onto a singular value decomposition (SVD) of a matrix using piecewise Legendre polynomial expansion.

!!! note

    The computation automatically uses optimized strategies: - For centrosymmetric kernels, specialized algorithms are employed - The working precision is adjusted to meet accuracy requirements

!!! note

    The returned object must be freed using [`spir_release_sve_result`](@ref) when no longer needed

# Arguments
* `k`: Pointer to the kernel object for which to compute SVE
* `epsilon`: Accuracy target for the basis. Determines: - The relative magnitude of included singular values - The accuracy of computed singular values and vectors
# Returns
A pointer to the newly created SVE result object containing the truncated singular value expansion, or NULL if creation fails
# See also
[`spir_release_sve_result`](@ref)
"""
function spir_sve_result_new(k, epsilon)
    ccall((:spir_sve_result_new, libsparseir), Ptr{spir_sve_result}, (Ptr{spir_kernel}, Cdouble), k, epsilon)
end

"""
    spir_kernel_domain(k, xmin, xmax, ymin, ymax)

Retrieves the domain boundaries of a kernel function.

This function obtains the domain boundaries (ranges) for both the x and y variables of the specified kernel function. The kernel domain is typically defined as a rectangle in the (x,y) plane.

# Arguments
* `k`: Pointer to the kernel object whose domain is to be retrieved
* `xmin`: Pointer to store the minimum value of the x-range
* `xmax`: Pointer to store the maximum value of the x-range
* `ymin`: Pointer to store the minimum value of the y-range
* `ymax`: Pointer to store the maximum value of the y-range
# Returns
0 on success, -1 on failure (if the kernel is invalid or an exception occurs)
"""
function spir_kernel_domain(k, xmin, xmax, ymin, ymax)
    ccall((:spir_kernel_domain, libsparseir), Cint, (Ptr{spir_kernel}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}), k, xmin, xmax, ymin, ymax)
end

"""
    spir_kernel_matrix(k, x, nx, y, ny, out)

Takes a kernel `k`, an array `x` of size `nx`, an array `y` of size `ny` and an array `out` of size `nx * ny`. On exit, set `out[ix*ny + iy] = K(x[ix], y[iy])`.
"""
function spir_kernel_matrix(k, x, nx, y, ny, out)
    ccall((:spir_kernel_matrix, libsparseir), Cint, (Ptr{spir_kernel}, Ptr{Cdouble}, Cint, Ptr{Cdouble}, Cint, Ptr{Cdouble}), k, x, nx, y, ny, out)
end

function spir_fermionic_finite_temp_basis_get_size(b, size)
    ccall((:spir_fermionic_finite_temp_basis_get_size, libsparseir), Cint, (Ptr{spir_fermionic_finite_temp_basis}, Ptr{Cint}), b, size)
end

function spir_bosonic_finite_temp_basis_get_size(b, size)
    ccall((:spir_bosonic_finite_temp_basis_get_size, libsparseir), Cint, (Ptr{spir_bosonic_finite_temp_basis}, Ptr{Cint}), b, size)
end

"""
    spir_fermionic_tau_sampling_new(b)

Creates a new fermionic tau sampling object for sparse sampling in imaginary time.

Constructs a sampling object that allows transformation between the IR basis and a set of sampling points in imaginary time (τ). The sampling points are automatically chosen as the extrema of the highest-order basis function in imaginary time, which provides near-optimal conditioning for the given basis size.

!!! note

    The sampling points are chosen to optimize numerical stability and accuracy

!!! note

    The sampling matrix is automatically factorized using SVD for efficient transformations

!!! note

    The returned object must be freed using [`spir_release_sampling`](@ref) when no longer needed

# Arguments
* `b`: Pointer to a fermionic finite temperature basis object
# Returns
A pointer to the newly created sampling object, or NULL if creation fails
# See also
[`spir_release_sampling`](@ref)
"""
function spir_fermionic_tau_sampling_new(b)
    ccall((:spir_fermionic_tau_sampling_new, libsparseir), Ptr{spir_sampling}, (Ptr{spir_fermionic_finite_temp_basis},), b)
end

"""
    spir_fermionic_matsubara_sampling_new(b)

Creates a new fermionic Matsubara sampling object for sparse sampling in Matsubara frequencies.

Constructs a sampling object that allows transformation between the IR basis and a set of sampling points in Matsubara frequencies (iωn). The sampling points are automatically chosen as the (discrete) extrema of the highest-order basis function in Matsubara frequencies, which provides near-optimal conditioning for the given basis size.

For fermionic Matsubara frequencies, the sampling points are odd integers: iωn = (2n + 1)π/β, where n is an integer.

!!! note

    The sampling points are chosen to optimize numerical stability and accuracy

!!! note

    The sampling matrix is automatically factorized using SVD for efficient transformations

!!! note

    For fermionic functions, the Matsubara frequencies are odd multiples of π/β, i.e. iωn = (2n + 1)π/β.

!!! note

    The returned object must be freed using [`spir_release_sampling`](@ref) when no longer needed

# Arguments
* `b`: Pointer to a fermionic finite temperature basis object
# Returns
A pointer to the newly created sampling object, or NULL if creation fails
# See also
[`spir_release_sampling`](@ref)
"""
function spir_fermionic_matsubara_sampling_new(b)
    ccall((:spir_fermionic_matsubara_sampling_new, libsparseir), Ptr{spir_sampling}, (Ptr{spir_fermionic_finite_temp_basis},), b)
end

"""
    spir_bosonic_tau_sampling_new(b)

Creates a new bosonic tau sampling object for sparse sampling in imaginary time.

Constructs a sampling object that allows transformation between the IR basis and a set of sampling points in imaginary time (τ). The sampling points are automatically chosen as the extrema of the highest-order basis function in imaginary time, which provides near-optimal conditioning for the given basis size.

!!! note

    The sampling points are chosen to optimize numerical stability and accuracy

!!! note

    The sampling matrix is automatically factorized using SVD for efficient transformations

!!! note

    The returned object must be freed using [`spir_release_sampling`](@ref) when no longer needed

# Arguments
* `b`: Pointer to a bosonic finite temperature basis object
# Returns
A pointer to the newly created sampling object, or NULL if creation fails
# See also
[`spir_release_sampling`](@ref)
"""
function spir_bosonic_tau_sampling_new(b)
    ccall((:spir_bosonic_tau_sampling_new, libsparseir), Ptr{spir_sampling}, (Ptr{spir_bosonic_finite_temp_basis},), b)
end

"""
    spir_bosonic_matsubara_sampling_new(b)

Creates a new bosonic Matsubara sampling object for sparse sampling in Matsubara frequencies.

Constructs a sampling object that allows transformation between the IR basis and a set of sampling points in Matsubara frequencies (iωn). The sampling points are automatically chosen as the (discrete) extrema of the highest-order basis function in Matsubara frequencies, which provides near-optimal conditioning for the given basis size.

For bosonic Matsubara frequencies, the sampling points are even integers: iωn = 2nπ/β, where n is an integer.

!!! note

    The sampling points are chosen to optimize numerical stability and accuracy

!!! note

    The sampling matrix is automatically factorized using SVD for efficient transformations

!!! note

    For bosonic functions, the Matsubara frequencies are even multiples of π/β

!!! note

    The returned object must be freed using [`spir_release_sampling`](@ref) when no longer needed

# Arguments
* `b`: Pointer to a bosonic finite temperature basis object
# Returns
A pointer to the newly created sampling object, or NULL if creation fails
# See also
[`spir_release_sampling`](@ref)
"""
function spir_bosonic_matsubara_sampling_new(b)
    ccall((:spir_bosonic_matsubara_sampling_new, libsparseir), Ptr{spir_sampling}, (Ptr{spir_bosonic_finite_temp_basis},), b)
end

"""
    spir_fermionic_dlr_new(b)

Creates a new fermionic Discrete Lehmann Representation (DLR).

This function implements a variant of the discrete Lehmann representation (DLR). Unlike the IR which uses truncated singular value expansion of the analytic continuation kernel K, the DLR is based on a "sketching" of K. The resulting basis is a linear combination of discrete set of poles on the real-frequency axis, continued to the imaginary-frequency axis:

G(iν) = ∑ a[i] / (iν - w[i]) for i = 1, 2, ..., L

where: - a[i] are the expansion coefficients - w[i] are the poles on the real axis - iν are the fermionic Matsubara frequencies

!!! note

    The poles on the real-frequency axis are selected based on the zeros of the IR basis functions on the real axis

!!! note

    The returned object must be freed using [`spir_release_fermionic_dlr`](@ref) when no longer needed

!!! warning

    This implementation uses a heuristic approach for pole selection, which differs from the original DLR method that uses rank-revealing decomposition

# Arguments
* `b`: Pointer to a fermionic finite temperature basis object
# Returns
A pointer to the newly created DLR object, or NULL if creation fails
# See also
[`spir_release_fermionic_dlr`](@ref), [`spir_fermionic_dlr_new_with_poles`](@ref)
"""
function spir_fermionic_dlr_new(b)
    ccall((:spir_fermionic_dlr_new, libsparseir), Ptr{spir_fermionic_dlr}, (Ptr{spir_fermionic_finite_temp_basis},), b)
end

"""
    spir_fermionic_dlr_new_with_poles(b, npoles, poles)

Creates a new fermionic Discrete Lehmann Representation (DLR) with custom poles.

This function creates a fermionic DLR using a set of user-specified poles on the real-frequency axis. The DLR represents Green's functions as a sum of poles:

G(iν) = ∑ a[i] / (iν - w[i]) for i = 1, 2, ..., npoles

where w[i] are the specified poles and a[i] are the expansion coefficients.

!!! note

    This function allows for more control over the pole selection compared to the automatic pole selection in [`spir_fermionic_dlr_new`](@ref)

# Arguments
* `b`: Pointer to a fermionic finite temperature basis object
* `npoles`: Number of poles to use in the representation
* `poles`: Array of pole locations on the real-frequency axis
# Returns
A pointer to the newly created DLR object with custom poles, or NULL if creation fails
# See also
[`spir_fermionic_dlr_new`](@ref), [`spir_release_fermionic_dlr`](@ref)
"""
function spir_fermionic_dlr_new_with_poles(b, npoles, poles)
    ccall((:spir_fermionic_dlr_new_with_poles, libsparseir), Ptr{spir_fermionic_dlr}, (Ptr{spir_fermionic_finite_temp_basis}, Cint, Ptr{Cdouble}), b, npoles, poles)
end

"""
    spir_bosonic_dlr_new(b)

Creates a new bosonic Discrete Lehmann Representation (DLR).

This function implements a variant of the discrete Lehmann representation (DLR). Unlike the IR which uses truncated singular value expansion of the analytic continuation kernel K, the DLR is based on a "sketching" of K. The resulting basis is a linear combination of discrete set of poles on the real-frequency axis, continued to the imaginary-frequency axis:

G(iωn) = ∑ a[i] / (iωn - w[i]) for i = 1, 2, ..., L

where: - a[i] are the expansion coefficients - w[i] are the poles on the real axis - iωn are the bosonic Matsubara frequencies (even multiples of π/β)

!!! note

    The poles on the real-frequency axis are selected based on the zeros of the IR basis functions on the real axis

!!! note

    The returned object must be freed using [`spir_release_bosonic_dlr`](@ref) when no longer needed

!!! warning

    This implementation uses a heuristic approach for pole selection, which differs from the original DLR method that uses rank-revealing decomposition

# Arguments
* `b`: Pointer to a bosonic finite temperature basis object
# Returns
A pointer to the newly created DLR object, or NULL if creation fails
# See also
[`spir_release_bosonic_dlr`](@ref), [`spir_bosonic_dlr_new_with_poles`](@ref)
"""
function spir_bosonic_dlr_new(b)
    ccall((:spir_bosonic_dlr_new, libsparseir), Ptr{spir_bosonic_dlr}, (Ptr{spir_bosonic_finite_temp_basis},), b)
end

"""
    spir_bosonic_dlr_new_with_poles(b, npoles, poles)

Creates a new bosonic Discrete Lehmann Representation (DLR) with custom poles.

This function creates a bosonic DLR using a set of user-specified poles on the real-frequency axis. The DLR represents correlation functions as a sum of poles:

G(iωn) = ∑ a[i] / (iωn - w[i]) for i = 1, 2, ..., npoles

where w[i] are the specified poles and a[i] are the expansion coefficients.

!!! note

    This function allows for more control over the pole selection compared to the automatic pole selection in [`spir_bosonic_dlr_new`](@ref)

# Arguments
* `b`: Pointer to a bosonic finite temperature basis object
* `npoles`: Number of poles to use in the representation
* `poles`: Array of pole locations on the real-frequency axis
# Returns
A pointer to the newly created DLR object with custom poles, or NULL if creation fails
# See also
[`spir_bosonic_dlr_new`](@ref), [`spir_release_bosonic_dlr`](@ref)
"""
function spir_bosonic_dlr_new_with_poles(b, npoles, poles)
    ccall((:spir_bosonic_dlr_new_with_poles, libsparseir), Ptr{spir_bosonic_dlr}, (Ptr{spir_bosonic_finite_temp_basis}, Cint, Ptr{Cdouble}), b, npoles, poles)
end

"""
    spir_sampling_eval_dd(s, order, ndim, input_dims, target_dim, input, out)

Evaluates basis coefficients at sampling points (double to double version).

Transforms basis coefficients to values at sampling points, where both input and output are real (double precision) values. The operation can be performed along any dimension of a multidimensional array.

!!! note

    For optimal performance, the target dimension should be either the first (0) or the last (ndim-1) dimension to avoid large temporary array allocations

!!! note

    The output array must be pre-allocated with the correct size

!!! note

    The input and output arrays must be contiguous in memory

# Arguments
* `s`: Pointer to the sampling object
* `order`: Memory layout order (SPIR\\_ORDER\\_ROW\\_MAJOR or SPIR\\_ORDER\\_COLUMN\\_MAJOR)
* `ndim`: Number of dimensions in the input/output arrays
* `input_dims`: Array of dimension sizes
* `target_dim`: Target dimension for the transformation (0-based)
* `input`: Input array of basis coefficients
* `out`: Output array for the evaluated values at sampling points
# Returns
0 on success, non-zero on failure
# See also
[`spir_sampling_eval_dz`](@ref), [`spir_sampling_eval_zz`](@ref)
"""
function spir_sampling_eval_dd(s, order, ndim, input_dims, target_dim, input, out)
    ccall((:spir_sampling_eval_dd, libsparseir), Cint, (Ptr{spir_sampling}, int32_t, Int32, Ptr{Int32}, Int32, Ptr{Cdouble}, Ptr{Cdouble}), s, order, ndim, input_dims, target_dim, input, out)
end

"""
    spir_sampling_eval_dz(s, order, ndim, input_dims, target_dim, input, out)

Evaluates basis coefficients at sampling points (double to complex version).

Transforms basis coefficients to values at sampling points, where input is real (double precision) and output is complex (double precision) values. The operation can be performed along any dimension of a multidimensional array.

!!! note

    For optimal performance, the target dimension should be either the first (0) or the last (ndim-1) dimension to avoid large temporary array allocations

!!! note

    The output array must be pre-allocated with the correct size

!!! note

    The input and output arrays must be contiguous in memory

!!! note

    Complex numbers are stored as pairs of consecutive double values (real, imag)

# Arguments
* `s`: Pointer to the sampling object
* `order`: Memory layout order (SPIR\\_ORDER\\_ROW\\_MAJOR or SPIR\\_ORDER\\_COLUMN\\_MAJOR)
* `ndim`: Number of dimensions in the input/output arrays
* `input_dims`: Array of dimension sizes
* `target_dim`: Target dimension for the transformation (0-based)
* `input`: Input array of real basis coefficients
* `out`: Output array for the evaluated complex values at sampling points
# Returns
0 on success, non-zero on failure
# See also
[`spir_sampling_eval_dd`](@ref), [`spir_sampling_eval_zz`](@ref)
"""
function spir_sampling_eval_dz(s, order, ndim, input_dims, target_dim, input, out)
    ccall((:spir_sampling_eval_dz, libsparseir), Cint, (Ptr{spir_sampling}, int32_t, Int32, Ptr{Int32}, Int32, Ptr{Cdouble}, Ptr{c_complex}), s, order, ndim, input_dims, target_dim, input, out)
end

function spir_sampling_eval_zz(s, order, ndim, input_dims, target_dim, input, out)
    ccall((:spir_sampling_eval_zz, libsparseir), Cint, (Ptr{spir_sampling}, int32_t, Int32, Ptr{Int32}, Int32, Ptr{c_complex}, Ptr{c_complex}), s, order, ndim, input_dims, target_dim, input, out)
end

"""
    spir_sampling_fit_dd(s, order, ndim, input_dims, target_dim, input, out)

Fits values at sampling points to basis coefficients (double to double version).

Transforms values at sampling points back to basis coefficients, where both input and output are real (double precision) values. The operation can be performed along any dimension of a multidimensional array.

!!! note

    For optimal performance, the target dimension should be either the first (0) or the last (ndim-1) dimension to avoid large temporary array allocations

!!! note

    The output array must be pre-allocated with the correct size

!!! note

    The input and output arrays must be contiguous in memory

!!! note

    This function performs the inverse operation of [`spir_sampling_eval_dd`](@ref)

# Arguments
* `s`: Pointer to the sampling object
* `order`: Memory layout order (SPIR\\_ORDER\\_ROW\\_MAJOR or SPIR\\_ORDER\\_COLUMN\\_MAJOR)
* `ndim`: Number of dimensions in the input/output arrays
* `input_dims`: Array of dimension sizes
* `target_dim`: Target dimension for the transformation (0-based)
* `input`: Input array of values at sampling points
* `out`: Output array for the fitted basis coefficients
# Returns
SPIR\\_COMPUTATION\\_SUCCESS on success, non-zero on failure
# See also
[`spir_sampling_eval_dd`](@ref), [`spir_sampling_fit_zz`](@ref)
"""
function spir_sampling_fit_dd(s, order, ndim, input_dims, target_dim, input, out)
    ccall((:spir_sampling_fit_dd, libsparseir), Cint, (Ptr{spir_sampling}, int32_t, Int32, Ptr{Int32}, Int32, Ptr{Cdouble}, Ptr{Cdouble}), s, order, ndim, input_dims, target_dim, input, out)
end

"""
    spir_sampling_fit_zz(s, order, ndim, input_dims, target_dim, input, out)

Fits values at sampling points to basis coefficients (complex to complex version).

Transforms values at sampling points back to basis coefficients, where both input and output are complex (double precision) values. The operation can be performed along any dimension of a multidimensional array.

!!! note

    For optimal performance, the target dimension should be either the first (0) or the last (ndim-1) dimension to avoid large temporary array allocations

!!! note

    The output array must be pre-allocated with the correct size

!!! note

    The input and output arrays must be contiguous in memory

!!! note

    Complex numbers are stored as pairs of consecutive double values (real, imag)

!!! note

    This function performs the inverse operation of [`spir_sampling_eval_zz`](@ref)

# Arguments
* `s`: Pointer to the sampling object
* `order`: Memory layout order (SPIR\\_ORDER\\_ROW\\_MAJOR or SPIR\\_ORDER\\_COLUMN\\_MAJOR)
* `ndim`: Number of dimensions in the input/output arrays
* `input_dims`: Array of dimension sizes
* `target_dim`: Target dimension for the transformation (0-based)
* `input`: Input array of complex values at sampling points
* `out`: Output array for the fitted complex basis coefficients
# Returns
SPIR\\_COMPUTATION\\_SUCCESS on success, non-zero on failure
# See also
[`spir_sampling_eval_zz`](@ref), [`spir_sampling_fit_dd`](@ref)
"""
function spir_sampling_fit_zz(s, order, ndim, input_dims, target_dim, input, out)
    ccall((:spir_sampling_fit_zz, libsparseir), Cint, (Ptr{spir_sampling}, int32_t, Int32, Ptr{Int32}, Int32, Ptr{c_complex}, Ptr{c_complex}), s, order, ndim, input_dims, target_dim, input, out)
end

"""
    spir_bosonic_dlr_fitmat_rows(dlr)

Gets the number of rows in the fitting matrix of a bosonic DLR.

This function returns the number of rows in the fitting matrix of the specified bosonic Discrete Lehmann Representation (DLR). The fitting matrix is used to transform between the DLR representation and values at sampling points.

!!! note

    The fitting matrix dimensions determine the size of valid input/output arrays for transformations involving this DLR object

# Arguments
* `dlr`: Pointer to the bosonic DLR object
# Returns
The number of rows in the fitting matrix, or SPIR\\_GET\\_IMPL\\_FAILED if the DLR object is invalid
# See also
[`spir_bosonic_dlr_fitmat_cols`](@ref), [`spir_bosonic_dlr_from_IR`](@ref), [`spir_bosonic_dlr_to_IR`](@ref)
"""
function spir_bosonic_dlr_fitmat_rows(dlr)
    ccall((:spir_bosonic_dlr_fitmat_rows, libsparseir), Cint, (Ptr{spir_bosonic_dlr},), dlr)
end

"""
    spir_bosonic_dlr_fitmat_cols(dlr)

Gets the number of columns in the fitting matrix of a bosonic DLR.

This function returns the number of columns in the fitting matrix of the specified bosonic Discrete Lehmann Representation (DLR). The fitting matrix is used to transform between the DLR representation and values at sampling points.

!!! note

    The fitting matrix dimensions determine the size of valid input/output arrays for transformations involving this DLR object

# Arguments
* `dlr`: Pointer to the bosonic DLR object
# Returns
The number of columns in the fitting matrix, or SPIR\\_GET\\_IMPL\\_FAILED if the DLR object is invalid
# See also
[`spir_bosonic_dlr_fitmat_rows`](@ref), [`spir_bosonic_dlr_from_IR`](@ref), [`spir_bosonic_dlr_to_IR`](@ref)
"""
function spir_bosonic_dlr_fitmat_cols(dlr)
    ccall((:spir_bosonic_dlr_fitmat_cols, libsparseir), Cint, (Ptr{spir_bosonic_dlr},), dlr)
end

"""
    spir_fermionic_dlr_fitmat_rows(dlr)

Gets the number of rows in the fitting matrix of a fermionic DLR.

This function returns the number of rows in the fitting matrix of the specified fermionic Discrete Lehmann Representation (DLR). The fitting matrix is used to transform between the DLR representation and values at sampling points.

!!! note

    The fitting matrix dimensions determine the size of valid input/output arrays for transformations involving this DLR object

# Arguments
* `dlr`: Pointer to the fermionic DLR object
# Returns
The number of rows in the fitting matrix, or SPIR\\_GET\\_IMPL\\_FAILED if the DLR object is invalid
# See also
[`spir_fermionic_dlr_fitmat_cols`](@ref), [`spir_fermionic_dlr_from_IR`](@ref), [`spir_fermionic_dlr_to_IR`](@ref)
"""
function spir_fermionic_dlr_fitmat_rows(dlr)
    ccall((:spir_fermionic_dlr_fitmat_rows, libsparseir), Cint, (Ptr{spir_fermionic_dlr},), dlr)
end

"""
    spir_fermionic_dlr_fitmat_cols(dlr)

Gets the number of columns in the fitting matrix of a fermionic DLR.

This function returns the number of columns in the fitting matrix of the specified fermionic Discrete Lehmann Representation (DLR). The fitting matrix is used to transform between the DLR representation and values at sampling points.

!!! note

    The fitting matrix dimensions determine the size of valid input/output arrays for transformations involving this DLR object

# Arguments
* `dlr`: Pointer to the fermionic DLR object
# Returns
The number of columns in the fitting matrix, or SPIR\\_GET\\_IMPL\\_FAILED if the DLR object is invalid
# See also
[`spir_fermionic_dlr_fitmat_rows`](@ref), [`spir_fermionic_dlr_from_IR`](@ref), [`spir_fermionic_dlr_to_IR`](@ref)
"""
function spir_fermionic_dlr_fitmat_cols(dlr)
    ccall((:spir_fermionic_dlr_fitmat_cols, libsparseir), Cint, (Ptr{spir_fermionic_dlr},), dlr)
end

"""
    spir_fermionic_dlr_from_IR(dlr, order, ndim, input_dims, input, out)

Transforms a given input array from the Imaginary Frequency (IR) representation to the Fermionic Discrete Lehmann Representation (DLR) using the specified DLR object.

!!! note

    The input and output arrays must be allocated with sufficient memory. The size of the input and output arrays should match the dimensions specified. The order type determines the memory layout of the input and output arrays. The function assumes that the input array is in the specified order type. The output array will be in the specified order type.

# Arguments
* `dlr`: Pointer to the fermionic DLR object
* `order`: Order type (C or Fortran)
* `ndim`: Number of dimensions
* `input_dims`: Array of dimensions
* `input`: Input coefficients array in IR representation
* `out`: Output array in DLR representation
# Returns
0 on success, or a negative value if an error occurred
# See also
[`spir_fermionic_dlr_to_IR`](@ref), [`spir_fermionic_dlr_fitmat_rows`](@ref), [`spir_fermionic_dlr_fitmat_cols`](@ref)
"""
function spir_fermionic_dlr_from_IR(dlr, order, ndim, input_dims, input, out)
    ccall((:spir_fermionic_dlr_from_IR, libsparseir), Cint, (Ptr{spir_fermionic_dlr}, int32_t, Int32, Ptr{Int32}, Ptr{Cdouble}, Ptr{Cdouble}), dlr, order, ndim, input_dims, input, out)
end

"""
    spir_bosonic_dlr_from_IR(dlr, order, ndim, input_dims, input, out)

Transforms coefficients from IR basis to bosonic DLR representation.

This function converts expansion coefficients from the Intermediate Representation (IR) basis to the Discrete Lehmann Representation (DLR). The transformation is performed by solving a linear system using the fitting matrix:

g\\_DLR = matrix \\ g\\_IR

where: - g\\_DLR are the coefficients in the DLR basis - g\\_IR are the coefficients in the IR basis - matrix is the SVD-factorized transformation matrix

!!! note

    The output array must be pre-allocated with the correct size

!!! note

    The input and output arrays must be contiguous in memory

!!! note

    This function is specifically for bosonic (symmetric) Green's functions

!!! note

    The transformation preserves the numerical properties of the representation

!!! note

    The transformation involves solving a linear system, which may be computationally more intensive than the forward transformation

# Arguments
* `dlr`: Pointer to the bosonic DLR object
* `order`: Memory layout order (SPIR\\_ORDER\\_ROW\\_MAJOR or SPIR\\_ORDER\\_COLUMN\\_MAJOR)
* `ndim`: Number of dimensions in the input/output arrays
* `input_dims`: Array of dimension sizes
* `input`: Input array of IR coefficients (double precision)
* `out`: Output array for the DLR coefficients (double precision)
# Returns
SPIR\\_COMPUTATION\\_SUCCESS on success, SPIR\\_GET\\_IMPL\\_FAILED on failure (if the DLR object is invalid or an error occurs)
# See also
[`spir_bosonic_dlr_to_IR`](@ref), [`spir_fermionic_dlr_from_IR`](@ref)
"""
function spir_bosonic_dlr_from_IR(dlr, order, ndim, input_dims, input, out)
    ccall((:spir_bosonic_dlr_from_IR, libsparseir), Cint, (Ptr{spir_bosonic_dlr}, int32_t, Int32, Ptr{Int32}, Ptr{Cdouble}, Ptr{Cdouble}), dlr, order, ndim, input_dims, input, out)
end

"""
    spir_bosonic_dlr_to_IR(dlr, order, ndim, input_dims, input, out)

Transforms coefficients from DLR basis to bosonic IR representation.

This function converts expansion coefficients from the Discrete Lehmann Representation (DLR) basis to the Intermediate Representation (IR) basis. The transformation is performed using the fitting matrix:

g\\_IR = fitmat * g\\_DLR

where: - g\\_IR are the coefficients in the IR basis - g\\_DLR are the coefficients in the DLR basis - fitmat is the transformation matrix

!!! note

    The output array must be pre-allocated with the correct size

!!! note

    The input and output arrays must be contiguous in memory

!!! note

    This function is specifically for bosonic (symmetric) Green's functions

!!! note

    The transformation is a direct matrix multiplication, which is typically faster than the inverse transformation

# Arguments
* `dlr`: Pointer to the bosonic DLR object
* `order`: Memory layout order (SPIR\\_ORDER\\_ROW\\_MAJOR or SPIR\\_ORDER\\_COLUMN\\_MAJOR)
* `ndim`: Number of dimensions in the input/output arrays
* `input_dims`: Array of dimension sizes
* `input`: Input array of DLR coefficients (double precision)
* `out`: Output array for the IR coefficients (double precision)
# Returns
SPIR\\_COMPUTATION\\_SUCCESS on success, SPIR\\_GET\\_IMPL\\_FAILED on failure (if the DLR object is invalid or an error occurs)
# See also
[`spir_bosonic_dlr_from_IR`](@ref), [`spir_fermionic_dlr_to_IR`](@ref)
"""
function spir_bosonic_dlr_to_IR(dlr, order, ndim, input_dims, input, out)
    ccall((:spir_bosonic_dlr_to_IR, libsparseir), Cint, (Ptr{spir_bosonic_dlr}, int32_t, Int32, Ptr{Int32}, Ptr{Cdouble}, Ptr{Cdouble}), dlr, order, ndim, input_dims, input, out)
end

"""
    spir_fermionic_dlr_to_IR(dlr, order, ndim, input_dims, input, out)

Transforms coefficients from DLR basis to fermionic IR representation.

This function converts expansion coefficients from the Discrete Lehmann Representation (DLR) basis to the Intermediate Representation (IR) basis. The transformation is performed using the fitting matrix:

g\\_IR = fitmat * g\\_DLR

where: - g\\_IR are the coefficients in the IR basis - g\\_DLR are the coefficients in the DLR basis - fitmat is the transformation matrix

!!! note

    The output array must be pre-allocated with the correct size

!!! note

    The input and output arrays must be contiguous in memory

!!! note

    This function is specifically for fermionic Green's functions

!!! note

    The transformation is a direct matrix multiplication, which is typically faster than the inverse transformation

# Arguments
* `dlr`: Pointer to the fermionic DLR object
* `order`: Memory layout order (SPIR\\_ORDER\\_ROW\\_MAJOR or SPIR\\_ORDER\\_COLUMN\\_MAJOR)
* `ndim`: Number of dimensions in the input/output arrays
* `input_dims`: Array of dimension sizes
* `input`: Input array of DLR coefficients (double precision)
* `out`: Output array for the IR coefficients (double precision)
# Returns
SPIR\\_COMPUTATION\\_SUCCESS on success, SPIR\\_GET\\_IMPL\\_FAILED on failure (if the DLR object is invalid or an error occurs)
# See also
[`spir_fermionic_dlr_from_IR`](@ref), [`spir_bosonic_dlr_to_IR`](@ref)
"""
function spir_fermionic_dlr_to_IR(dlr, order, ndim, input_dims, input, out)
    ccall((:spir_fermionic_dlr_to_IR, libsparseir), Cint, (Ptr{spir_fermionic_dlr}, int32_t, Int32, Ptr{Int32}, Ptr{Cdouble}, Ptr{Cdouble}), dlr, order, ndim, input_dims, input, out)
end

function spir_fermionic_finite_temp_basis_new(beta, omega_max, epsilon)
    ccall((:spir_fermionic_finite_temp_basis_new, libsparseir), Ptr{spir_fermionic_finite_temp_basis}, (Cdouble, Cdouble, Cdouble), beta, omega_max, epsilon)
end

"""
    spir_bosonic_finite_temp_basis_new(beta, omega_max, epsilon)

Creates a new bosonic finite temperature IR basis.

For a continuation kernel K from real frequencies, ω ∈ [-ωmax, ωmax], to imaginary time, τ ∈ [0, β], this function creates an intermediate representation (IR) basis that stores the truncated singular value expansion:

K(τ, ω) ≈ ∑ u[l](τ) * s[l] * v[l](ω) for l = 1, 2, 3, ...

where: - u[l](τ) are IR basis functions on the imaginary time axis (stored as piecewise Legendre polynomials) - s[l] are singular values of the continuation kernel - v[l](ω) are IR basis functions on the real frequency axis (stored as piecewise Legendre polynomials)

!!! note

    The basis includes both imaginary time and Matsubara frequency representations

!!! note

    For Matsubara frequencies, bosonic basis uses even numbers (2n)

!!! note

    The returned object must be freed using [`spir_release_bosonic_finite_temp_basis`](@ref) when no longer needed

# Arguments
* `beta`: Inverse temperature β (must be positive)
* `omega_max`: Frequency cutoff ωmax (must be non-negative)
* `epsilon`: Accuracy target for the basis
# Returns
A pointer to the newly created bosonic finite temperature basis object, or NULL if creation fails
# See also
[`spir_release_bosonic_finite_temp_basis`](@ref)
"""
function spir_bosonic_finite_temp_basis_new(beta, omega_max, epsilon)
    ccall((:spir_bosonic_finite_temp_basis_new, libsparseir), Ptr{spir_bosonic_finite_temp_basis}, (Cdouble, Cdouble, Cdouble), beta, omega_max, epsilon)
end

"""
    spir_fermionic_finite_temp_basis_new_with_sve(beta, omega_max, k, sve)

Creates a new fermionic finite temperature IR basis using a pre-computed SVE result.

This function creates a fermionic intermediate representation (IR) basis using a pre-computed singular value expansion (SVE) result. This allows for reusing an existing SVE computation, which can be more efficient than recomputing it.

!!! note

    Using a pre-computed SVE can significantly improve performance when creating multiple basis objects with the same kernel

# Arguments
* `beta`: Inverse temperature β (must be positive)
* `omega_max`: Frequency cutoff ωmax (must be non-negative)
* `k`: Pointer to the kernel object used for the basis construction
* `sve`: Pointer to a pre-computed SVE result for the kernel
# Returns
A pointer to the newly created fermionic finite temperature basis object, or NULL if creation fails (invalid inputs or exception occurs)
# See also
[`spir_sve_result_new`](@ref), [`spir_release_fermionic_finite_temp_basis`](@ref)
"""
function spir_fermionic_finite_temp_basis_new_with_sve(beta, omega_max, k, sve)
    ccall((:spir_fermionic_finite_temp_basis_new_with_sve, libsparseir), Ptr{spir_fermionic_finite_temp_basis}, (Cdouble, Cdouble, Ptr{spir_kernel}, Ptr{spir_sve_result}), beta, omega_max, k, sve)
end

"""
    spir_bosonic_finite_temp_basis_new_with_sve(beta, omega_max, k, sve)

Creates a new bosonic finite temperature IR basis using a pre-computed SVE result.

This function creates a bosonic intermediate representation (IR) basis using a pre-computed singular value expansion (SVE) result. This allows for reusing an existing SVE computation, which can be more efficient than recomputing it.

!!! note

    Using a pre-computed SVE can significantly improve performance when creating multiple basis objects with the same kernel

# Arguments
* `beta`: Inverse temperature β (must be positive)
* `omega_max`: Frequency cutoff ωmax (must be non-negative)
* `k`: Pointer to the kernel object used for the basis construction
* `sve`: Pointer to a pre-computed SVE result for the kernel
# Returns
A pointer to the newly created bosonic finite temperature basis object, or NULL if creation fails (invalid inputs or exception occurs)
# See also
[`spir_sve_result_new`](@ref), [`spir_release_bosonic_finite_temp_basis`](@ref)
"""
function spir_bosonic_finite_temp_basis_new_with_sve(beta, omega_max, k, sve)
    ccall((:spir_bosonic_finite_temp_basis_new_with_sve, libsparseir), Ptr{spir_bosonic_finite_temp_basis}, (Cdouble, Cdouble, Ptr{spir_kernel}, Ptr{spir_sve_result}), beta, omega_max, k, sve)
end

"""
    spir_fermionic_finite_temp_basis_get_u(b)

Gets the basis functions of a fermionic finite temperature basis.

This function returns a polynomial vector containing the basis functions of the specified fermionic finite temperature basis. The basis functions are stored as piecewise Legendre polynomials.

!!! note

    The returned polynomial vector must be freed using spir\\_release\\_polyvector when no longer needed

# Arguments
* `b`: Pointer to the fermionic finite temperature basis object
# Returns
A pointer to the polynomial vector containing the basis functions, or NULL if the basis object is invalid
# See also
spir\\_release\\_polyvector
"""
function spir_fermionic_finite_temp_basis_get_u(b)
    ccall((:spir_fermionic_finite_temp_basis_get_u, libsparseir), Ptr{spir_continuous_functions}, (Ptr{spir_fermionic_finite_temp_basis},), b)
end

"""
    spir_fermionic_finite_temp_basis_get_v(b)

Gets the basis functions of a fermionic finite temperature basis.

This function returns a polynomial vector containing the basis functions of the specified fermionic finite temperature basis. The basis functions are stored as piecewise Legendre polynomials.

!!! note

    The returned polynomial vector must be freed using spir\\_release\\_polyvector when no longer needed

# Arguments
* `b`: Pointer to the fermionic finite temperature basis object
# Returns
A pointer to the polynomial vector containing the basis functions, or NULL if the basis object is invalid
# See also
spir\\_release\\_polyvector
"""
function spir_fermionic_finite_temp_basis_get_v(b)
    ccall((:spir_fermionic_finite_temp_basis_get_v, libsparseir), Ptr{spir_continuous_functions}, (Ptr{spir_fermionic_finite_temp_basis},), b)
end

"""
    spir_fermionic_finite_temp_basis_get_uhat(b)

Gets the basis functions of a fermionic finite temperature basis in Matsubara frequency domain.

This function returns a polynomial vector containing the basis functions of the specified fermionic finite temperature basis in Matsubara frequency domain.

!!! note

    The returned object must be freed using [`spir_release_matsubara_funcs`](@ref) when no longer needed

# Arguments
* `b`: Pointer to the fermionic finite temperature basis object
# Returns
A pointer to the object containing the basis functions, or NULL if the basis object is invalid
# See also
[`spir_release_matsubara_funcs`](@ref)
"""
function spir_fermionic_finite_temp_basis_get_uhat(b)
    ccall((:spir_fermionic_finite_temp_basis_get_uhat, libsparseir), Ptr{spir_matsubara_funcs}, (Ptr{spir_fermionic_finite_temp_basis},), b)
end

"""
    spir_bosonic_finite_temp_basis_get_u(b)

Gets the basis functions of a bosonic finite temperature basis.

This function returns a polynomial vector containing the basis functions of the specified bosonic finite temperature basis. The basis functions are stored as piecewise Legendre polynomials.

!!! note

    The returned polynomial vector must be freed using spir\\_release\\_polyvector when no longer needed

# Arguments
* `b`: Pointer to the bosonic finite temperature basis object
# Returns
A pointer to the polynomial vector containing the basis functions, or NULL if the basis object is invalid
# See also
spir\\_release\\_polyvector
"""
function spir_bosonic_finite_temp_basis_get_u(b)
    ccall((:spir_bosonic_finite_temp_basis_get_u, libsparseir), Ptr{spir_continuous_functions}, (Ptr{spir_bosonic_finite_temp_basis},), b)
end

"""
    spir_bosonic_finite_temp_basis_get_v(b)

Gets the basis functions of a bosonic finite temperature basis on the real frequency axis.

This function returns a polynomial vector containing the basis functions of the specified bosonic finite temperature basis on the real frequency axis. The basis functions are stored as piecewise Legendre polynomials.

!!! note

    The returned polynomial vector must be freed using spir\\_release\\_polyvector when no longer needed

# Arguments
* `b`: Pointer to the bosonic finite temperature basis object
# Returns
A pointer to the polynomial vector containing the basis functions, or NULL if the basis object is invalid
# See also
spir\\_release\\_polyvector
"""
function spir_bosonic_finite_temp_basis_get_v(b)
    ccall((:spir_bosonic_finite_temp_basis_get_v, libsparseir), Ptr{spir_continuous_functions}, (Ptr{spir_bosonic_finite_temp_basis},), b)
end

"""
    spir_bosonic_finite_temp_basis_get_uhat(b)

Gets the basis functions of a bosonic finite temperature basis in Matsubara frequency domain.

This function returns a polynomial vector containing the basis functions of the specified bosonic finite temperature basis in Matsubara frequency domain.

!!! note

    The returned object must be freed using [`spir_release_matsubara_funcs`](@ref) when no longer needed

# Arguments
* `b`: Pointer to the bosonic finite temperature basis object
# Returns
A pointer to the object containing the basis functions, or NULL if the basis object is invalid
# See also
[`spir_release_matsubara_funcs`](@ref)
"""
function spir_bosonic_finite_temp_basis_get_uhat(b)
    ccall((:spir_bosonic_finite_temp_basis_get_uhat, libsparseir), Ptr{spir_matsubara_funcs}, (Ptr{spir_bosonic_finite_temp_basis},), b)
end

"""
    spir_evaluate_continuous_functions(uv, x, out)

Evaluates basis functions at a single point in the imaginary-time domain or the real frequency domain.

This function evaluates all basis functions contained in a polynomial vector at a specified point x. The values of each basis function at x are stored in the output array. The output array out[j] contains the value of the j-th basis function evaluated at x.

!!! note

    The output array must be pre-allocated with sufficient size to store all basis function values

# Arguments
* `uv`: Pointer to the polynomial vector containing the basis functions
* `x`: Point at which to evaluate the basis functions
* `out`: Pre-allocated array to store the evaluation results. Must have size >= n\\_basis
# Returns
SPIR\\_COMPUTATION\\_SUCCESS on success, error code on failure
"""
function spir_evaluate_continuous_functions(uv, x, out)
    ccall((:spir_evaluate_continuous_functions, libsparseir), Int32, (Ptr{spir_continuous_functions}, Cdouble, Ptr{Cdouble}), uv, x, out)
end

"""
    spir_funcs_batch_eval_matsu(uiw, order, num_freqs, matsubara_freq_indices, out)

Evaluates basis functions at multiple Matsubara frequencies.

This function evaluates all basis functions contained in a Matsubara basis functions object at the specified Matsubara frequency indices. The values of each basis function at each frequency are stored in the output array.

!!! note

    The output array must be pre-allocated with sufficient size to store all basis function values at all requested frequencies. Indices n correspond to ωn = nπ/β, where n are odd for fermionic frequencies and even for bosonic frequencies.

# Arguments
* `uiw`: Pointer to the Matsubara basis functions object
* `order`: Memory layout order (SPIR\\_ORDER\\_ROW\\_MAJOR or SPIR\\_ORDER\\_COLUMN\\_MAJOR)
* `num_freqs`: Number of Matsubara frequencies at which to evaluate
* `matsubara_freq_indices`: Array of Matsubara frequency indices
* `out`: Pre-allocated array to store the evaluation results. The results are stored as a 2D array of size n\\_basis x num\\_freqs.
# Returns
SPIR\\_COMPUTATION\\_SUCCESS on success, error code on failure
"""
function spir_funcs_batch_eval_matsu(uiw, order, num_freqs, matsubara_freq_indices, out)
    ccall((:spir_funcs_batch_eval_matsu, libsparseir), Int32, (Ptr{spir_matsubara_funcs}, int32_t, Int32, Ptr{Int32}, Ptr{c_complex}), uiw, order, num_freqs, matsubara_freq_indices, out)
end

"""
    spir_sampling_get_npoints(s, num_points)

Gets the number of sampling points in a sampling object.

This function returns the number of sampling points used in the specified sampling object. This number is needed to allocate arrays of the correct size when retrieving the actual sampling points.

# Arguments
* `s`: Pointer to the sampling object
* `num_points`: Pointer to store the number of sampling points
# Returns
SPIR\\_COMPUTATION\\_SUCCESS on success, SPIR\\_GET\\_IMPL\\_FAILED if the sampling object is invalid
# See also
[`spir_sampling_get_taus`](@ref), [`spir_sampling_get_matsus`](@ref)
"""
function spir_sampling_get_npoints(s, num_points)
    ccall((:spir_sampling_get_npoints, libsparseir), Cint, (Ptr{spir_sampling}, Ptr{Cint}), s, num_points)
end

"""
    spir_sampling_get_taus(s, points)

Gets the imaginary time sampling points.

This function fills the provided array with the imaginary time (τ) sampling points used in the specified sampling object. The array must be pre-allocated with sufficient size (use [`spir_sampling_get_npoints`](@ref) to determine the required size).

!!! note

    The array must be pre-allocated with size >= [`spir_sampling_get_npoints`](@ref)(s)

# Arguments
* `s`: Pointer to the sampling object
* `points`: Pre-allocated array to store the τ sampling points
# Returns
SPIR\\_COMPUTATION\\_SUCCESS on success SPIR\\_GET\\_IMPL\\_FAILED if s is invalid SPIR\\_NOT\\_SUPPORTED if the sampling object is not for τ sampling
# See also
[`spir_sampling_get_npoints`](@ref)
"""
function spir_sampling_get_taus(s, points)
    ccall((:spir_sampling_get_taus, libsparseir), Cint, (Ptr{spir_sampling}, Ptr{Cdouble}), s, points)
end

"""
    spir_sampling_get_matsus(s, points)

Gets the Matsubara frequency sampling points.

This function fills the provided array with the Matsubara frequency indices (n) used in the specified sampling object. The actual Matsubara frequencies are ωn = (2n + 1)π/β for fermionic case and ωn = 2nπ/β for bosonic case. The array must be pre-allocated with sufficient size (use [`spir_sampling_get_npoints`](@ref) to determine the required size).

!!! note

    The array must be pre-allocated with size >= [`spir_sampling_get_npoints`](@ref)(s)

!!! note

    For fermionic case, the indices n give frequencies ωn = (2n + 1)π/β

!!! note

    For bosonic case, the indices n give frequencies ωn = 2nπ/β

# Arguments
* `s`: Pointer to the sampling object
* `points`: Pre-allocated array to store the Matsubara frequency indices
# Returns
SPIR\\_COMPUTATION\\_SUCCESS on success SPIR\\_GET\\_IMPL\\_FAILED if s is invalid SPIR\\_NOT\\_SUPPORTED if the sampling object is not for Matsubara sampling
# See also
[`spir_sampling_get_npoints`](@ref)
"""
function spir_sampling_get_matsus(s, points)
    ccall((:spir_sampling_get_matsus, libsparseir), Cint, (Ptr{spir_sampling}, Ptr{Cint}), s, points)
end

@cenum int32_t::Int32 begin
    SPIR_COMPUTATION_SUCCESS = 0
    SPIR_GET_IMPL_FAILED = -1
    SPIR_INVALID_DIMENSION = -2
    SPIR_INPUT_DIMENSION_MISMATCH = -3
    SPIR_OUTPUT_DIMENSION_MISMATCH = -4
    SPIR_NOT_SUPPORTED = -5
    SPIR_INVALID_ARGUMENT = -6
    SPIR_INTERNAL_ERROR = -7
end

const SPARSEIR_VERSION_MAJOR = 0

const SPARSEIR_VERSION_MINOR = 1

const SPARSEIR_VERSION_PATCH = 0

# exports
const PREFIXES = ["spir_"]
for name in names(@__MODULE__; all=true), prefix in PREFIXES
    if startswith(string(name), prefix)
        @eval export $name
    end
end

end # module
