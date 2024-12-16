using Gridap
using Random

function run(L,n,order,Pe,Δt,t₀,T,τᵈkₒ)

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
    β = 0.0001
    aᵘ(u,v) = ∫( eₕ * ( ∇(u)⊙∇(v) ) + β * ( u⋅v ) )dΩ
    bᵘ(v) = ∫( Pe * ( ∇(eₕ) ⋅ v ) )dΩ
    aᵘ,bᵘ
  end

  function transport_problem(uₕ,eₕ,Δt,τᵈkₒ)
    m(e,ε) = ∫( (1/Δt)*(e*ε) )dΩ
    d(e,ε) = ∫( ∇(e)⋅∇(ε) )dΩ
    c(e,ε) =  ∫( (uₕ⋅∇(e))*ε + tr(∇(uₕ))*(e*ε) )dΩ
    r(e,ε) = ∫( τᵈkₒ*(e*ε) )dΩ
    γ = 0.001
    s(e,ε) = ∫( γ*(e*e*e*ε) )dΩ
    l(ε) = ∫( τᵈkₒ*ε )dΩ
    aᵐ(e,ε) = m(e,ε) + c(e,ε) + r(e,ε) + d(e,ε)
    bᵐ(ε) = m(eₕ,ε) + l(ε) - s(eₕ,ε)
    aᵐ,bᵐ
  end

  i = 0
  t = t₀

  tol = 1e-8

  while t < T + tol

    @info "Time step $i, time $t and time step $Δt"

    aᵘ,bᵘ = viscous_flow_problem(Pe,eₕ)
    opᵘ = AffineFEOperator(aᵘ,bᵘ,Uᵘ,Vᵘ)
    uₕ = solve(opᵘ)
    
    aᵐ,bᵐ = transport_problem(uₕ,eₕ,Δt,τᵈkₒ)
    opᵐ = AffineFEOperator(aᵐ,bᵐ,Uᵐ,Vᵐ)
    eₕ = solve(opᵐ)
   
    writevtk(Ω,"test_$i",cellfields=["uₕ"=>uₕ,"eₕ"=>eₕ])    

    i = i + 1
    t = t + Δt

  end

end

Pe = 1.0
Δt = 0.001
t₀ = 0.0
T = 1.0
τᵈkₒ = 1.0

L = 1
n = 10
order = 1

run(L,n,order,Pe,Δt,t₀,T,τᵈkₒ)