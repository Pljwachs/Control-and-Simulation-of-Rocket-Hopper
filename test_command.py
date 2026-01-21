# importing libraries
import serial  # Library for serial communication with python - install command: pip install pyserial
import time  # time tracking library
from mpccontroller import MPCController
import numpy as np


# teensy loop time
loop_cycle_time = 1/60 # 60 measurement aquistions per second; can be adjusted to max of 66Hz
control_duration = 10 # in [s] --> your anticipated flight duration / control event duration

### establishing serial connection
# you need to figure out the serial port ID of your laptop; it might be different from the predefined
ser = serial.Serial('/dev/cu.usbserial-1130', baudrate=57600)

# setup/start commands for the hopper
# resetting error status
ser.write(b'0')

# activating valve
ser.write(b'1')

# setting the reply mode --> sending the input back yes or no
# r = reply 
# n = no reply
ser.write(b'n') # donÄt change this

# setting the failure mode
# F = no failure
# f = failure
ser.write(b'F') # don't change this

## States
#1. Hose / 2. Fitting /3. Main Valve/4. Nozzle
# Initial pressure (Pa)
p_s=11e5                  # p_s pressure supply: at the inlet of the hose   
p1_t0=p_s                 # inlet of the fitting
p2_t0=p_s                 # inlet of the valve
p3_t0=1e5                 # inlet of the nozzle:  assume initial constant air pressure
p3_max_range=11*1e5       # maximum input command for the valve

p3_u_t0=1e5
pe_t0=1e5


# Initial mass flow rate (kg/s)
md1_t0=0
md2_t0=0
md3_t0=0
md4_t0=0

state_t0=np.array([p1_t0,md1_t0,p2_t0,md2_t0,p3_t0,md3_t0,p3_u_t0,md4_t0])
state_t=np.zeros((len(state_t0),len(time)))


# Initial density (assuming nitrogen at T=273K)
rho_1_t0 = 1.2336 * p_s / 1e5
rho_2_t0 = 1.2336 * p1_t0 / 1e5
rho_3_t0 = 1.2336 * p2_t0 / 1e5


rho_t0=np.array([rho_1_t0,rho_2_t0,rho_3_t0])
rho_t=np.zeros((len(rho_t0),len(time)))
###########################################################
# Motion (dynamic system)
x_t0=0.0
v_t0=0.0
a_t0=0.0

x_target=3.0         # height target [m] (change to test different height)

motion_t0=np.array([x_t0,v_t0])
motion_t=np.zeros((len(motion_t0),len(time)))

###########################################################
# System & environmental parameters

## Hose
H_l=8                      # length [m]
H_d=19.3e-3                # inner diameter[m]
L_1=8                      #  [m]
A_1=np.pi*(H_d/2)**2       # [m^2]
V_1=L_1*A_1                # [m^3] A_1*H_l  


## Fitting
F_l=49e-3                      # length [m]
F_d=11.9e-3                    # diameter [m]
A_2=np.pi*(F_d/2)**2           #[m^2]
V_2=A_2*F_l                    #[m^3] #A_2*F_l


## Main Valve
Kv=4.8           
Open_t=825e-3    # opening time [s] (825)
Close_t=1700e-3  # closing time[s]
T_3=273          # [K]


## Nozzle
Th_d=9e-3                        # throat diameter [m]
Th_exit_d=11e-3                  # exit diameter [m]
A_throat=np.pi*(Th_d/2)**2       # [m^2]
A_e=np.pi*(Th_exit_d/2)**2       # [m^2]
epsilon=A_e/A_throat             # expansion ratio
T_4=273                          # [k]

###########################################################

## mass
m_hop=2.8                       # mass of hopper [kg] (3.5)
m_hose=1.0                      # mass of hose   [kg] (1.0)
prev_pose=x_t0                  # for hose dynamic modeling
k_hose=4.8                        # spring constant [N/m]
F_RR=10                         # rolling resistence [N]

## constant
R_gas=8314         # universal gas constant [J/kmol*K]
M=28.013           # nitrogen gas  [kg/kmol]
gamma=1.4          # ratio of specific heats (diatomic gas)
g=9.81             # [m/s^2]


###########################################################################################
## Calculate the basic force and coefficients

md4n_term=A_throat/np.sqrt(T_4*R_gas/(M*gamma)) * np.power((gamma+1)/2, -(gamma+1)/(2*(gamma-1)))
ve_term=(2*gamma*R_gas*T_4)/(gamma-1)/M

ve_suggest_val=float(np.sqrt(ve_term * (1 - np.power(0.1615, (gamma - 1) / gamma))))
print("ve_suggest_value:", ve_suggest_val)

thrust_req= m_hop*(g)+k_hose*x_target
pressure_req=thrust_req/(ve_suggest_val*md4n_term)

print("Basic force need to sustain horizontal at the equilibrium is: ", thrust_req ," [N]")
print("Basic pressure command to reach equilibrium at target", x_target, ":", pressure_req, " [Pa]")

######################################################################
## For MPC parameter (cost function)
Q=np.array([[10.0,0.0],[0.0,10.0]])      
R=np.array([0.01])
Qf=10*Q

N=20                                            # horizon
freq=20

# initial input
u0=np.array([p3_u_t0]).reshape(-1,1)            # control input p3_u
z0=np.array([md4_t0,pe_t0]).reshape(-1,1)       # algebraic parameter  initial value: md4n pe       

#target state: xs / initial state x0
xs=np.array([x_target,0.0]).reshape(-1,1)                   
x0=np.array([motion_t0[0],motion_t0[1]]).reshape(-1,1)
mpc=MPCController(motion_t0,u0,z0,Q,R,Qf,xs,freq,N)  # Hz 
# Side Note: If the MPC frequency is the same as the simulation frequency can reduce strange oscillation, but need to consider the cost

mpc.setup()


# -------------------------------------------------------------------------
### main loop
main_timer = time.time()                            # control timer --> duration of entire control event / flight duration

while True:
    loop_timer = time.time()                        # loop timer --> the time it takes for one control loop is either equivalnt to the loop_cycle_time or slower
    
    # signal coming from teensy
    #--------------------------------
    raw_data = serial.readline().decode().strip()   # old line
    raw_data = ser.readline().decode().strip()      # <- correct line
    #--------------------------------s
    raw_data = raw_data.split(':')                  # spliting the message message
    
    teensy_time = int(raw_data[0][1:])              # time in ms since teensy power on
    acceleration = float(raw_data[1]) - 9.80665     # float in [m/s^2] --> when Hopper at rest, it shows + 9.80665 m/s^2
    position = float(raw_data[2])                   # float in [m]
    pressure = int(raw_data[3])                     # int as 12 bit signal --> 0 corresponds to 0barg; 4095 correponds to 10barg
    
    '''
    INSERT YOUR CONTROL LAW HERE

    
    
    action = control_law(.....) # action must be a 12 bit signal

    
    '''   

    p3_u=mpc.compute_action(x0=np.array(motion).reshape(-1,1),z0=np.array(alg_state).reshape(-1,1),xs=np.array([x_target,0.0]).reshape(-1,1),u0=np.array(p3_u))

    # control input which is going to be sent to teensy
    action = f'<1:{action}>'
    
    # wait until loop time has passed
    while time.time() - loop_timer < loop_cycle_time:
        continue
    
    # send control input / action to teensy
    ser.write(action.encode())
   
    # stop the control event once, control duration is reached
    #------------------------- old line
    if time.time() - main_timer < control_duration:
        break
    #-------------------------- new line
    if time.time() - main_timer > control_duration:
        break
    #--------------------------

ser.write('<1:1000>'.encode())  # reduce thrust to decrease altitude
time.sleep(1)

ser.write('<1:0>'.encode())  # shut off valve
time.sleep(0.1)
ser.write('<1:0>'.encode())  # shut off valve
time.sleep(0.1)
ser.write('<1:0>'.encode())  # shut off valve

# close serial connection
ser.close()