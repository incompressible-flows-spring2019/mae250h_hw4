export RK, RKParams
export RK4, RK31, Euler


struct RKParams{N}
  c::Vector{Float64}
  a::Matrix{Float64}
end

const RK4 = RKParams{4}([0.5,0.5,1.0,1.0],
                        [1/2  0    0   0
                         0    1/2  0   0
                         0    0    1   0
                         1/6  1/3  1/3 1/6])

const RK31 = RKParams{3}([0.5, 1.0, 1.0],
                      [1/2        0        0
                       √3/3 (3-√3)/3        0
                       (3+√3)/6    -√3/3 (3+√3)/6])

const Euler = RKParams{1}([1.0],ones(1,1))


"""
    RK(u,Δt,f;[rk::RKParams=RK4])

Construct an integrator to advance a system of the form

du/dt = f(u,t)

The resulting integrator will advance the state `u` by one time step, `Δt`.

# Arguments

- `u` : example of state vector data
- `Δt` : time-step size
- `f` : operator acting on type `u` and `t` and returning `u`
"""
struct RK{NS,TU}

  # time step size
  Δt :: Number

  rk :: RKParams

  f

  # cache
  qᵢ :: TU
  w :: Vector{TU}

end


function (::Type{RK})(u::TU,Δt::Real,rhs;
                          rk::RKParams{NS}=RK4) where {TU,NS}

    # check for methods for r₁ and r₂
    if hasmethod(rhs,(TU,Real))
        f = rhs
    else
        error("No valid operator for f supplied")
    end

    # allocate cache
    qᵢ = deepcopy(u)
    w = [deepcopy(u) for i = 1:NS-1]

    # fuse the time step size into the coefficients for some cost savings
    rkdt = deepcopy(rk)
    rkdt.a .*= Δt
    rkdt.c .*= Δt

    rksys = RK{NS,TU}(Δt,rkdt,f,qᵢ,w)

    return rksys
end

function Base.show(io::IO, scheme::RK{NS,TU}) where {NS,TU}
    println(io, "Order-$NS RK integator with")
    println(io, "   State of type $TU")
    println(io, "   Time step size $(scheme.Δt)")
end

function (scheme::RK{NS,TU})(t::Real,u::TU) where {NS,TU}

  # some shorthands
  Δt = scheme.Δt
  rk = scheme.rk
  f  = scheme.f
  qᵢ = scheme.qᵢ
  w  = scheme.w

  # Remember that each of the coefficients includes the time step size
  i = 1
  tᵢ₊₁ = t
  qᵢ .= u

  if NS > 1
    # first stage, i = 1
    w[i] .= rk.a[i,i].*f(u,tᵢ₊₁) # gᵢ

    u .= qᵢ .+ w[i]
    tᵢ₊₁ = t + rk.c[i]


    # stages 2 through NS-1
    for i = 2:NS-1
      w[i-1] ./= rk.a[i-1,i-1]
      w[i] .= rk.a[i,i].*f(u,tᵢ₊₁)

      u .= qᵢ .+ w[i] # r₁
      for j = 1:i-1
        u .+= rk.a[i,j]*w[j]
      end
      tᵢ₊₁ = t + rk.c[i]

    end
    i = NS
    w[i-1] ./= rk.a[i-1,i-1]
  end

  # final stage (assembly)
  u .= qᵢ .+ rk.a[i,i].*f(u,tᵢ₊₁)
  for j = 1:i-1
    u .+= rk.a[i,j]*w[j]
  end
  t = t + rk.c[i]


  return t, u

end
