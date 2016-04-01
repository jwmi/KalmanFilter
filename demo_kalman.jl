# Demo of Kalman filter and smoother on a toy problem 
# involving the height of a projectile. 

# Copyright (c) 2016 Jeffrey W. Miller
# This software is released under the MIT License.
# If you use this software in your research, please cite:
# 
#   Jeffrey W. Miller (2016). Lecture Notes on Advanced Stochastic Modeling. Duke University, Durham, NC.

# The "Punkin Chunkin" contest is an annual engineering challenge in which
# teams use trebuchets, catapults, and air cannons in a competition to see
# who can hurl a pumpkin the farthest possible distance. The current record
# as of this writing is 4694.7 feet (1430.9 meters), nearly one mile.
# https://www.punkinchunkin.com/results/current-records.html
#
# Let's suppose we instrument a pumpkin with an accelerometer and
# a GPS, and we use an air cannon to hurl a pumpkin straight up
# into the air. We can use a Kalman filter and smoother to estimate 
# the trajectory of the pumpkin over time. For simplicity, let's suppose
# that the GPS and accelerometer give measurements synchronously,
# and that the range of the accelerometer is unlimited (although in practice
# there would be some upper and lower bounds, and the resulting truncation
# would need to be modeled). Let's consider only the vertical dimension (height).


include("kalman.jl")

module demo_kalman
using Kalman
can_plot = (Pkg.installed("PyPlot")!=nothing)  # check if PyPlot is installed
if can_plot; using PyPlot; end
srand(1) # set seed of random number generator

# Simulation settings
m = 5 # mass (kg) (mass of the pumpkin)
R = 0.15 # radius (m) (radius of the pumpkin)
g = 9.8 # gravity (m/s^2)
dt = 1e-4 # temporal resolution (s) of physics simulation
T = 20 # length of time (s) to run physics simulation
T_display = T # length of time (s) to display in plots
interval = 100 # number of time steps between measurements
sigma_a = 0.1*g # 1e-3*g # std dev of acceleration measurement error
sigma_p = 50.0 # std dev of position measurement error
t_propel = 0.1 # length of time (s) of propulsion
F_propel = 2e4 # force (kg*m/s^2) of propulsion
sigma_wind = 5.0 # std dev of wind speed (m/s)

# Derived settings
start_index = round(Int, t_propel/dt)+1
n_steps = round(Int,T/dt) # number of time steps to run physics simulation
wind = sigma_wind*cumsum(randn(n_steps))./sqrt(1:n_steps)
wind_velocity(t) = wind[round(Int,t/dt)+1] # vertical wind velocity (m/s^2) at time t

# Physics
# Drag parameters
# (for simplicity, let's assume a constant density of air)
rho = 1.20 # density of air (kg/m^3) at T = 20 deg C, sea level.
A = pi*R^2 # projection area (of the pumpkin)
C_D = 0.47 # drag coefficient for a sphere
force_drag(v) = -sign(v)*v*v*0.5*rho*A*C_D
# External forces (propulsion)
force_external(t) = F_propel*(0 < t < t_propel)
# Force at time t if the current velocity is v
force(t,v) = force_external(t) + force_drag(v - wind_velocity(t)) - m*g

# Simulate trajectory with time step dt, for n_steps
function simulate_trajectory(force,dt,n_steps)
    # Initialize
    t = (0:dt:dt*(n_steps-1)) # time
    a = zeros(n_steps) # acceleration
    v = zeros(n_steps) # velocity
    p = zeros(n_steps) # position
    for i = 1:n_steps-1
        a[i] = force(t[i],v[i])/m
        v[i+1] = v[i] + a[i]*dt
        p[i+1] = p[i] + v[i]*dt + 0.5*a[i]*dt*dt
    end
    return t,a,v,p
end

# Simulate measurements
function simulate_measurements(a,v,p,I)
    n = length(I)
    a_m = a[I] + sigma_a*randn(n)
    p_m = p[I] + sigma_p*randn(n)
    return a_m,p_m
end

# Run physics simulation
t,a,v,p = simulate_trajectory(force,dt,n_steps)
I = (start_index:interval:n_steps) # indices at which to take measurements
a_m,p_m = simulate_measurements(a,v,p,I)

# Set up parameters of Kalman filter and smoother
dtm = dt*interval
x = [a_m p_m]'
mu_0 = [-500,400,20]
V_0 = diagm([50^2,40^2,2^2])
F2 = [1 0 0;
     dtm 1 0;
     0.5*dtm^2 dtm 1] # Second-order model
F = [1 0 0;
     dtm 1 0;
     0 dtm 1] # First-order model
H = [1 0 0;
     0 0 1]
sigma_a_model = 2
sigma_p_model = 75
R = diagm([sigma_a_model^2, sigma_p_model^2])
Q = diagm([100*dtm,dtm,dtm]) # initial guess

# Run Kalman filter and smoother
mu,V,P = Kalman.filter(x,mu_0,V_0,F,Q,H,R)
mu_h,V_h = Kalman.smoother(mu,V,P,F)
# # Refine estimate of Q and rerun
# Q = diagm(vec(mean((F*mu_h[:,1:end-1] - mu_h[:,2:end]).^2, 2)))
# println(diag(Q))
# mu,V,P = Kalman.filter(x,mu_0,V_0,F,Q,H,R)
# mu_h,V_h = Kalman.smoother(mu,V,P,F)
a_est = vec(mu_h[1,:]) # estimated acceleration
v_est = vec(mu_h[2,:]) # estimated velocity
p_est = vec(mu_h[3,:]) # estimated position

# Save inputs
# writedlm("x.txt",x')
# writedlm("true.txt",[a[I] v[I] p[I]])
# println("dtm = ",dtm)
# println("n = ",size(x,2))


# Compare simpler approach of using an IIR filter
f = 0.9
n = size(x,2)
u = zeros(n)
u[1] = x[2,1]
for i = 2:n
    u[i] = f*u[i-1] + (1-f)*x[2,i]
end

# Report RMS error
RMS_a = sqrt(mean((a[I] - a_est).^2))
RMS_v = sqrt(mean((v[I] - v_est).^2))
RMS_p = sqrt(mean((p[I] - p_est).^2))
println("RMS acceleration = ",RMS_a)
println("RMS velocity = ",RMS_v)
println("RMS position = ",RMS_p)
RMS_u = sqrt(mean((p[I] - u).^2))
println("RMS position (simple) = ",RMS_u)

# Display plots
if can_plot
    figure(1); clf(); hold(true)
    
    # Plot position (true, measured, and estimated)
    subplot(3,1,1); grid(true)
    title("Pumpkin tracking with Kalman filter")
    plot(t[I],p_m,"k.",markersize=.25)
    plot(t[I],p[I],"b",label="true")
    plot(t[I],p_est,"r--",label="estimated")
    #plot(t[I],u,"g--",label="simple")
    xlabel("time (s)")
    ylabel("position (m)")
    ylim(0,ylim()[2])
    xlim(0,T_display)
    
    # Plot velocity (true and estimated)
    subplot(3,1,2); grid(true)
    plot(t[I],v[I],"b",label="true")
    plot(t[I],v_est,"r--",label="estimated")
    xlabel("time (s)")
    ylabel("velocity (m/s)")
    xlim(0,T_display)
    
    # Plot acceleration (true, measured, and estimated)
    subplot(3,1,3); grid(true)
    plot(t[I],a_m,"k.",markersize=.2)
    plot(t[I],a[I],"b",label="true")
    plot(t[I],a_est,"r--",label="estimated")
    xlabel("time (s)")
    ylabel("accel (m/s^2)")
    legend(loc="lower right")
    xlim(0,T_display)
    
    #savefig("pumpkin.png",dpi=150)
end



end # module
