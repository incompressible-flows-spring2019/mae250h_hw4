export trisolve

"""
    trisolve(a,b,c,f::Vector{<:Real},sys_type::String) -> Vector{Real}

Solve a tridiagonal system, in which the matrix consists of elements `a`, `b`, and `c`,
on the subdiagonal, diagonal, and superdiagonal, respectively, and the right-hand
side of the system is given by vector `f`. The `sys_type` of system can be either `"circulant"`
or `"regular"`. The solution is returned as a vector the same size as `f`.

For `"circulant"` systems, `a`, `b` and `c` can only be scalars, since the diagonals
must each be uniform. For `"regular"` systems, these can each be vectors, but `b` must
be the same length as `f` and `a` and `c` should be one element shorter.
"""
function trisolve(a,b,c,f::Vector{<:Real},sys_type::String)
  # Note that `f` is set up to only accept vectors whose elements are of type Real
  # or some subtype of Real (e.g. Float64). That is what the <: does.

  # Return immediately with zeros if f is all zeros
  if all(f .== 0)
    return f
  end

  if sys_type == "circulant"

    # For circulant matrices, use this special form
    return circulant(a,b,c,f)

  elseif sys_type == "regular"

    # For regular matrices, use this special form
    return thomas(a,b,c,f)

  end

end

# ======  Circulant tridiagonal matrices ==============#

function circulant(a::Real,b::Real,c::Real,f::Vector{<:Real})
  # For circulant matrices with only allow a, b, c to be scalars

  M = length(f)

  # create a solution vector that is just a copy of f. We will operate on this
  # vector "in place". We are also converting it to complex type for use in the
  # fft
  x = ComplexF64.(f)

  m = 0:M-1
  # Get the eigenvalues of the matrix
  # In Julia, you can get special symbols like π by typing backslash, then pi,
  # then [TAB].
  lam = b .+ (a+c)*cos.(2π*m/M) .- im*(a-c)*sin.(2π*m/M)

  # x now temporarily holds the transform of the right-hand side
  FFTW.fft!(x)

  # Scale x by M to make it agree with what we usually call the discrete
  # Fourier transform
  x ./= M

  # now divide the transform of f by the eigenvalues of the matrix
  # This is a shorthand take each element of x, divide it by the corresponding
  # entry in lam, and put the result in x.
  x ./= lam

  # Now inverse transform the result and scale appropriately and return just
  # the real part of each element
  FFTW.ifft!(x)

  x .*= M # multiply by the vector length to get the "correct" form of IFFT

  return real.(x)

end

# ======  Regular tridiagonal matrices ==============#

# For scalar-valued a, b, c, call the vector-valued form. This ensures
# that we have only one function to write and debug
function thomas(a::Real,b::Real,c::Real,f::Vector{<:Real})
  n = length(f)
  ftype = eltype(f) # this is to ensure that a, b, c elements have same type as f
  return thomas(a*ones(ftype,n-1),
                b*ones(ftype,n),
                c*ones(ftype,n-1),f)
end

function thomas(a::Vector{<:Real},b::Vector{<:Real},c::Vector{<:Real},f::Vector{<:Real})
  # This is the main algorithm for regular circulant matrices

  # We will develop the solution x in place
  x = copy(f)

  M = length(f)

  if b[1] == 0
    error("First pivot is zero")
  end

  # Modify the first row of coefficients
  x[1] /= b[1]

  if M == 1
    # There is only one entry, so we are done!
    return x
  end

  ctmp = copy(c)
  
  ctmp[1] /= b[1]
  tmp1 = ctmp[1]
  tmp2 = x[1]
  for i = 2:M-1
    tmp = b[i] - a[i-1]*tmp1
    tmp1 = ctmp[i]/tmp
    tmp2 = (x[i]-a[i-1]*tmp2)/tmp
    ctmp[i] = tmp1
    x[i] = tmp2
  end
  x[M] = (x[M]-a[M-1]*x[M-1])/(b[M]-a[M-1]*ctmp[M-1])

  # Back substitute
  tmp = x[M]
  for i = M-1:-1:1
    tmp = x[i] - ctmp[i]*tmp
    x[i] = tmp
  end

  return x

end
