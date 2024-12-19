using Gridap
using Random
using Plots; pythonplot()

function run(L,n,order,Δt,t₀,T,Pe,τᵈkₒ)

  domain = (0,L)
  partition = (n)
  model = CartesianDiscreteModel(domain,partition;isperiodic=(true,))

  degree = 2*order
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)

  reffeᵘ = ReferenceFE(lagrangian,VectorValue{1,Float64},order)
  Vᵘ = TestFESpace(Ω,reffeᵘ)
  Uᵘ = TrialFESpace(Vᵘ)

  reffeᵐ = ReferenceFE(lagrangian,Float64,order)
  Vᵐ = TestFESpace(Ω,reffeᵐ)
  Uᵐ = TrialFESpace(Vᵐ)

  Vˡ = ConstantFESpace(model)
  Uˡ = TrialFESpace(Vˡ)

  Yᵈ = MultiFieldFESpace([Vᵐ,Vˡ])
  Xᵈ = MultiFieldFESpace([Uᵐ,Uˡ])

  Random.seed!(1234)
  _eʳ = 0.00001 * randn(Float64,num_free_dofs(Uᵐ))
  eʳ = FEFunction(Uᵐ,_eʳ)

  function initial_condition(eʳ)
    _rᵃ(u,v) = ∫( u*v )dΩ
    _rᵇ(v) = ∫( eʳ*v )dΩ
    _rᵘ(u,ℓ) = ∫( u*ℓ )dΩ
    rᵃ((u,l),(v,ℓ)) = _rᵃ(u,v) + _rᵘ(u,ℓ) + _rᵘ(v,l)
    rᵇ((v,ℓ)) = _rᵇ(v)
    rᵃ,rᵇ
  end

  rᵃ,rᵇ = initial_condition(eʳ)
  opᵈ = AffineFEOperator(rᵃ,rᵇ,Xᵈ,Yᵈ)
  _eₕ,_ = solve(opᵈ)
  _eₕ = 1.0 + _eₕ
  eₕ = interpolate_everywhere(_eₕ,Uᵐ)

  function viscous_flow_problem(Pe,eₕ)
    β = 1.0e-4 * Pe
    dξ(e) = 4*e / ( e*e + 1 )^2
    aᵘ(u,v) = ∫( 2.0 * ( ∇(u)⊙∇(v) ) + β * ( u⋅v ) )dΩ
    bᵘ(v) = ∫( Pe * ( dξ∘(eₕ) ) * ( ∇(eₕ) ⋅ v ) )dΩ
    aᵘ,bᵘ
  end

  function transport_problem(uₕ,eₕ,Δt,τᵈkₒ)
    m(e,ε) = ∫( (1/Δt)*(e*ε) )dΩ
    d(e,ε) = ∫( ∇(e)⋅∇(ε) )dΩ
    c(e,ε) =  ∫( (uₕ⋅∇(e))*ε + tr(∇(uₕ))*(e*ε) )dΩ
    r(e,ε) = ∫( τᵈkₒ*(e*ε) )dΩ
    l(ε) = ∫( τᵈkₒ*ε )dΩ
    aᵐ(e,ε) = m(e,ε) + c(e,ε) + r(e,ε) + d(e,ε)
    bᵐ(ε) = m(eₕ,ε) + l(ε)
    aᵐ,bᵐ
  end

  i = 0
  t = t₀

  nt = 1+Int(T/Δt)
  nx = n+1
  kymograph_u = zeros(nt,nx)
  kymograph_e = zeros(nt,nx)
  st = range(t₀,T,nt)
  sx = range(0.0,L,nx)

  tol = 1e-8

  while t < T + tol

    @info "Time step $i, time $t and time step $Δt"

    aᵘ,bᵘ = viscous_flow_problem(Pe,eₕ)
    opᵘ = AffineFEOperator(aᵘ,bᵘ,Uᵘ,Vᵘ)
    uₕ = solve(opᵘ)
    vuₓ = uₕ(Point.(sx))
    kymograph_u[1+i,:] = map(x -> x[1], vuₓ)

    aᵐ,bᵐ = transport_problem(uₕ,eₕ,Δt,τᵈkₒ)
    opᵐ = AffineFEOperator(aᵐ,bᵐ,Uᵐ,Vᵐ)
    eₕ = solve(opᵐ)
    kymograph_e[1+i,:] = eₕ(Point.(sx))

    i = i + 1
    t = t + Δt

  end

  plt = heatmap(sx,st,kymograph_u,c=:jet,yflip=true,yticks=(0.0:0.2:1.0,string.(0.0:0.2:1.0)),size=(600,600))
  savefig(plt,"plots/actomyosinmodel/u_Pe_$(trunc(Pe,digits=2))_r_$(trunc(τᵈkₒ,digits=2)).png")
  plt = heatmap(sx,st,kymograph_e,c=:jet,yflip=true,yticks=(0.0:0.2:1.0,string.(0.0:0.2:1.0)),size=(600,600))
  savefig(plt,"plots/actomyosinmodel/e_Pe_$(trunc(Pe,digits=2))_r_$(trunc(τᵈkₒ,digits=2)).png")

  maxe = maximum(kymograph_e[nt,:])
  if maxe < 1.0+1.0e-6
    maxe = 0.9
  end
  maximum(kymograph_u[nt,:]),maxe

end

Δt = 0.01
t₀ = 0.0
T = 1.0

L = 1
n = 100
order = 1

pts = 100
Pes = logrange(10,1000,pts)
τᵈkₒs = logrange(1,100,pts)

datau = zeros(pts,pts)
datae = zeros(pts,pts)

for (i,Pe) in enumerate(Pes)
  for (j,τᵈkₒ) in enumerate(τᵈkₒs)
    maxu,maxe = run(L,n,order,Δt,t₀,T,Pe,τᵈkₒ)
    datau[i,j] = maxu
    datae[i,j] = maxe
  end
end

plt = contour(τᵈkₒs,Pes,datae,xscale=:log10,yscale=:log10,
              size=(600,600),yticks=(10 .^(1:1:3),string.(10 .^(1:1:3))),
              xticks=(10 .^(0:1:2),string.(10 .^(0:1:2))),
              xlabel="τᵈkₒ",ylabel="Pe",
              levels=10, color=:blues, clabels=true, cbar=false, lw=1)
savefig(plt,"plots/maxe-actomyosinmodel.png")

plt = heatmap(τᵈkₒs,Pes,datau,xscale=:log10,yscale=:log10,
              size=(600,600),yticks=(10 .^(1:1:3),string.(10 .^(1:1:3))),
              xticks=(10 .^(0:1:2),string.(10 .^(0:1:2))),
              xlabel="τᵈkₒ",ylabel="Pe",
              color=:blues,cbar=true,colorbar_ticks=(0:10:60,string.(0:10:60)))
savefig(plt,"plots/maxu-actomyosinmodel.png")