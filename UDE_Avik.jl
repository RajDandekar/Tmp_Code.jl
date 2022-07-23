
cd("/Users/sreedath/Desktop/Julia Computing 2022 Job")
Pkg.activate(".")

using LinearAlgebra
using Flux, DiffEqFlux
using OrdinaryDiffEq
using JLD
using LaTeXStrings
using Plots
using Random

#=================================#
#= Setting the data scenario     =#
#=================================#
# Dictionary of our training data.
D = load("SIRHD_simple_true_normalize.jld")
data_N_days = length(D["S"])
train_N_days = Int64(data_N_days) # Days to run the simulation

d_train = Dict(
    "I" => D["I"][1:train_N_days],
    "R" => D["R"][1:train_N_days],
    "H" => D["H"][1:train_N_days],
    "D" => D["D"][1:train_N_days],
    )


#=============================#
#= Setting up initial values =#
#=============================#

S0 = 1.
u0 = [S0*0.99, S0*0.01, 0., 0., 0.]
tspan = (0.0, Float64(train_N_days))
datasize = train_N_days
t = range(tspan[1],tspan[2],length=datasize)

p0_vec = Float64[]

##Initializing the Neural Networks
NN1 = FastChain(FastDense(2,10,relu),FastDense(10,1))
p0 = initial_params(NN1)
append!(p0_vec, p0)

NN2 = FastChain(FastDense(1,10,relu),FastDense(10,1))
p0 = initial_params(NN2)
append!(p0_vec, p0)

NN3 = FastChain(FastDense(1,10,relu),FastDense(10,1))
p0 = initial_params(NN3)
append!(p0_vec, p0)

NN4 = FastChain(FastDense(1,10,relu),FastDense(10,1))
p0 = initial_params(NN4)
append!(p0_vec, p0)

NN5 = FastChain(FastDense(1,10,relu),FastDense(10,1))
p0 = initial_params(NN5)
append!(p0_vec, p0)

NN6 = FastChain(FastDense(1,10,relu),FastDense(10,1))
p0 = initial_params(NN6)
append!(p0_vec, p0)


function SIRHD_NN!(du, u, p, t)
    (S,I,R,H, D) = u

    NNSI = abs(NN1([S,I], p[1:41])[1])
    NNIR = abs(NN2([I], p[42:72])[1])
    NNID = abs(NN3([I], p[73:103])[1])
    NNIH = abs(NN4([I], p[104:134])[1])
    NNHR = abs(NN5([H], p[135:165])[1])
    NNHD = abs(NN6([H], p[166:196])[1])


    du[1] = dS = -NNSI
    du[2] = dI = NNSI - NNIR - NNID - NNIH
    du[3] = dR = NNIR + NNHR
    du[4] = dH = NNIH - NNHR - NNHD
    du[5] = dD = NNID + NNHD
end

prob = ODEProblem(SIRHD_NN!, u0, tspan, p0_vec)
sol = Array(solve(prob, Tsit5(), u0=u0, p=p0_vec, saveat=t, reltol = 1e-4))


function predict_adjoint(p)
  Array(solve(prob,Tsit5(), u0=u0, p=p, saveat=t, reltol = 1e-4 ))
end

#=======================================#
#= Instantiating ODE/loss function/etc =#
#= for training                        =#
#=======================================#
function loss_fn(p)
    prediction = predict_adjoint(p)

    loss =  sum( abs2, (d_train["I"] .- prediction[2,:])[2:end])
    loss += sum( abs2, (d_train["R"] .- prediction[3,:])[2:end])
    loss += sum( abs2, (d_train["H"] .- prediction[4,:])[2:end])
    loss += sum( abs2, (d_train["D"] .- prediction[5,:])[2:end])

    loss, prediction
end

loss_fn(p0_vec)

global Loss = []
global P = []

cb = function(p,l,pred)
  #display(l[1])
  push!(Loss, l[1])
  global P = append!(P, p)

   println(Loss[end])
  false
end


res = DiffEqFlux.sciml_train(loss_fn, p0_vec, ADAM(0.0001), maxiters = 30000, cb=cb)
