# 2D charge quantization 

using Cubature
using DifferentialEquations

# We first set hbar=1, mass=1, trap frequency=1. This sets the energy unit. Everything in the following is then dimensionless.
# Time is measured in inverse trap frequency, length is meausred in harmonic length, and energy is measured in hbar*omega.

eF=100.0      # Fermi energy, much larger than the trap frequency
st=0.2  # standard deviation for time, t is measured in inverse trap frequency
sx=2  # standard deviation for x, x is measured in harmonic length
T=0.1*eF  # temperature compared to Fermi energy
t1=pi*0.4 # time of the first pulse
t2=pi*0.8 # time of second pulse
t3=pi*1.2 # time of measurement
N=1 # pulse strength

# a function to be called in the integrand
function f(x,y,kx,ky,eF,T)
    E=(x^2+y^2+kx^2+ky^2)/2 # Total energy of a particle in a 2D harmonic trap
    return (1/(2*T^2))*sinh((E-eF)/T)/((1+cosh((E-eF)/T))^2)
end 

# Gauss distribution
function Gauss(z,mu,sigma)
    return exp(-(z-mu)^2/(2*sigma^2))/(sqrt(2*pi)*sigma)
end

# time-reversal equations of montion of particles in a Harmonic trap with kicks in each direction
function trajectory(du,u,p,t)
    q, k=u                                          # q, k are position and momentum 
    t0, st, sx, N=p                                    # t0 is the time of the kick, st, sx are standard deviations in time and space domain
    du[1]=-k                                        # dq/dt=-k
    du[2]=q-2*pi*Gauss(t,t0,st)*Gauss(q,0.0,sx)*N     # dk/dt=q-F, each pulse F has a Gaussian profile in space and time
end

# define integrand
function integrand(r)
    tau=r[1]
    t=r[2]
    x=r[3]
    y=r[4]
    kx=r[5]
    ky=r[6]
    ff=f(x,y,kx,ky,eF,T)
    ux0=[x,kx] # The next six lines are to compute xt and kt numerically, with x, k as the "initial" condition
    tspanx=(0,t3-tau) # As x, kx are variables at t3, time evolution is reversed
    px=[t3-t1,st,sx,N] # In the reversed time frame, kick happens at t3-t1
    probx=ODEProblem(trajectory,ux0,tspanx,px)
    solx=solve(probx,reltol=1e-4,save_everystep=false) # relative error of trajectory set to 1e-4
    xtau, kxtau=solx[end]
    uy0=[y,ky] # The next six lines are to compute xt and kt numerically, with y, ky as the "initial" condition
    tspany=(0,t3-t) # As y, ky are variables at t3, time evolution is reversed
    py=[t3-t2,st,sx,N] # In the reversed time frame, kick happens at t3-t2
    proby=ODEProblem(trajectory,uy0,tspany,py)
    soly=solve(proby,reltol=1e-4,save_everystep=false)
    yt, kyt=soly[end]
    gtau=Gauss(tau,t1,st)
    gt=Gauss(t,t2,st)
    gxtau=Gauss(xtau,0,sx)
    gyt=Gauss(yt,0,sx)
    return gtau*gt*gxtau*gyt*kxtau*kyt*ff*0.5*(sign(t-tau)+1)   # Expression for the integrand 
end


lim=2*sqrt(eF) # Integration limit. Here we put a cutoff for both space and momentum as the integrand goes to zeros in the region E>>eF
n1, error1 = hcubature(integrand,[0,0,0,0,-lim,-lim],[t3,t3,lim,lim,lim,lim],reltol=1e-3) # n1 is the extra charge in the first quadrant, error1 its estimated error
n3, error3 = hcubature(integrand,[0,0,0,0,-lim,-lim],[t3,t3,lim,lim,lim,lim],reltol=1e-3) # n3 is the extra charge in the third quadrant, error3 its estimated error
n=(n1+n3)/2 # We take the average of n1 and n3

println(n1)
println(n)

# Plots can be produced afterwards.