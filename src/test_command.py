# importing libraries
import serial  # Library for serial communication with python - install command: pip install pyserial
import time  # time tracking library
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

x_target=1.0        # height target [m]


## Nozzle
Th_d=9e-3                       # throat diameter [m]
Th_exit_d=11e-3                 # exit diameter [m]
A_throat=np.pi*(Th_d/2)**2      # area throat[m^2]
A_e=np.pi*(Th_exit_d/2)**2      # area exit nozzle[m^2]
epsilon=A_e/A_throat            # expansion ratio
T_4=273                         # [k]
###########################################################

## mass
m_hop=3.5       # mass of hopper [kg]
m_hose=1.0      # mass of hose   [kg]
k_hose=6.0        # variable hopper mass [N/m]
F_RR=10         # rolling resistence [N]

## constant
R=8314             # universal gas constant [J/kmol*K]
M=28.013           # nitrogen gas  [kg/kmol]
gamma=1.4          # ratio of specific heats (diatomic gas)
g=9.81             # [m/s^2]
###########################################################################################
## Calculate the basic force and coefficients

md4n_term=A_throat/np.sqrt(T_4*R/(M*gamma)) * np.power((gamma+1)/2, -(gamma+1)/(2*(gamma-1)))
ve_term=(2*gamma*R*T_4)/(gamma-1)/M

ve_suggest_val=float(np.sqrt(ve_term * (1 - np.power(0.1615, (gamma - 1) / gamma))))
print("ve_suggest_value:", ve_suggest_val)

thrust_req= m_hop*(g)+k_hose*x_target
pressure_req=thrust_req/(ve_suggest_val*md4n_term)

print("Basic force need to sustain horizontal at the equilibrium is: ", thrust_req ," [N]")
print("Basic pressure command to reach equilibrium at target", x_target, ":", pressure_req, " [Pa]")
###########################################################################################
## PID parameters for pressure command (gain has to be larger since the step size is really small)
kp=4.0                # another reference value 1.2 0.5 1.0
ki=0.8
kd=4.0

FF_term=pressure_req*0.70                
print("Feed-Forward term: ",FF_term)


###########################################################
# PID controller for the pressure command
def pressure_PID(x,h,kp,ki,kd, FF_term):
   global integral, lasterr
  
   err=x-x_target
   integral=integral+ki*err*h
   derr=(err-lasterr)/h

   PIDlaw=-(kp*err+kd*derr+ki*integral)+FF_term
   #print("error: ",err, "integral: ", integral, "derror: ",derr)

   if PIDlaw<1:
      output=1
   elif PIDlaw>11:
      output=11
   else:
      output=PIDlaw

   lasterr=err

   #print("pressure output: ",output)
   return output



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

    p3_u=pressure_PID(x,h,kp,ki,kd)*1e5+FF_term                          # control input  (introduce the time delay modify once apply the controller+ 
                                                                        # add feed forward term to combat gravity and improve stailization
    p3_u=np.clip(p3_u,min=1e5, max=11e5)                                  # restrict the pressure input 

    # control input which is going to be sent to teensy
    action = f'<1:{action}>'
    
    # wait until loop time has passed
    while time.time() - loop_timer < loop_cycle_time:
        continue
    
    # send control input / action to teensy
    ser.write(action.encode())
   
    # stop the control event once, control duration is reached
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