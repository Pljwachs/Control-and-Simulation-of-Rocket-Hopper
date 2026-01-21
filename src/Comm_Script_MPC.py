# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 14:36:14 2026

@author: felix
"""

# importing libraries
import serial
import time
import matplotlib.pyplot as plt
import numpy as np
from mpccontroller import MPCController

# teensy loop time
loop_cycle_time = 1/60 # 60 measurement aquistions per second; can be adjusted to max of 66Hz
control_duration = 10 # in [s] --> your anticipated flight duration / control event duration

log = np.zeros((5, int(control_duration/loop_cycle_time+1)))

### establishing serial connection
# you need to figure out the serial port ID of your laptop; it might be different from the predefined
ser = serial.Serial('/dev/ttyUSB0', baudrate=57600)

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

x_target = 2
integral = 0
lasterr = 0
prev_position = 0.0
Th_d=9e-3                        # throat diameter [m]
A_throat=np.pi*(Th_d/2)**2      #[m^2]
T_4=273                        #[k]
prev_pos = 0
eqbm_count = 0
eqbm = 0

## constant
R_gas=8314          # universal gas constant [J/kmol*K]
M=28.013           # nitrogen gas  [kg/kmol]
gamma=1.4          # ratio of specific heats (diatomic gas)

# teensy loop time
loop_cycle_time = 1/60
control_duration = 10 #12
start_time = time.time()
prev_time = 0
prev_pressure = 0

print("Starting control loop...")
message_count = 0
received_count = 0
sent_count = 0

## For MPC parameter (cost function)
Q=np.array([[10.0,0.0],[0.0,10.0]])      
R=np.array([0.01])
Qf=100*Q

N=20                                          # horizon
freq=10

# initial input
u0=np.array([1e5]).reshape(-1,1)            # control input p3_u
z0=np.array([0,1e5]).reshape(-1,1)      # algebraic parameter  initial value: md4n pe       
motion_t0=np.array([0,0])  

#target state: xs / initial state x0
xs=np.array([x_target,0.0]).reshape(-1,1)                   
x0=np.array([motion_t0[0],motion_t0[1]]).reshape(-1,1)
mpc=MPCController(motion_t0,u0,z0,Q,R,Qf,xs,freq,N)  # Hz 
# Side Note: If the MPC frequency is the same as the simulation frequency can reduce strange oscillation, but need to consider the cost

# some notes regard the horizon (N) and frequency
# N: the higher the horizon-> more easy to converge -> but also very costly
# freq: the higher the inner-loop rate-> the controller react more quickly to disturbance or model mismatch and better tracking performance -> but also reduce 
# the prediction duration, if the prediction duration is not enough, it will also not converge 

mpc.setup()

Me = 1.849  # from your calculation

# Open file for writing
#folder_path = 'Test_Dummy' #Also change the traj value
#file_path = folder_path + '/simulation_data.txt'

# -------------------------------------------------------------------------
### main loop
main_timer = time.time()                            # control timer --> duration of entire control event / flight duration
counter = 0
while True:
    loop_timer = time.time()                        # loop timer --> the time it takes for one control loop is either equivalnt to the loop_cycle_time or slower
    
    # signal coming from teensy
    raw_data = ser.readline().decode().strip()   # reading the data comming from the teensy, decoding the data and removing white spaces
    raw_data = raw_data.split(':')                  # spliting the message message
    
    teensy_time = int(raw_data[0][1:])              # time in ms since teensy power on
    acceleration = float(raw_data[1]) - 9.80665     # float in [m/s^2] --> when Hopper at rest, it shows + 9.80665 m/s^2
    position = float(raw_data[2])                   # float in [m]
    pressure = int(raw_data[3])                     # int as 12 bit signal --> 0 corresponds to 0barg; 4095 correponds to 10barg
    
    
    '''INSERT YOUR CONTROL LAW HERE '''  

    velocity = (position - prev_position) / ((teensy_time - prev_time)/1000)
    md4=A_throat*pressure/(np.sqrt(T_4*R_gas/(M*gamma)))*np.power((gamma+1)/2,-(gamma+1)/(2*(gamma-1)))  # m_choked (at choked condition)
    xt = [position, velocity]

    pe = pressure / ((1 + ((gamma-1)/2 * Me**2))**(gamma/(gamma-1)))
    alg_state = np.array([md4, pe])

    action = mpc.compute_action(x0=np.array(xt).reshape(-1,1),z0=np.array(alg_state).reshape(-1,1),
                           xs=np.array([x_target,0.0]).reshape(-1,1),u0=np.array(prev_position))
    action= int((((action/1e5)-1)/10)*4095)

    prev_time = teensy_time
    prev_pressure = pressure
    prev_position = position

    log[:5, counter] = [teensy_time, acceleration, position, pressure, action]
     
    # control input which is going to be sent to teensy
    action = f'<1:{action}>'
    
    # wait until loop time has passed
    while time.time() - loop_timer < loop_cycle_time:
        continue
    
    # send control input / action to teensy
    ser.write(action.encode())
   
    counter += 1 

    message_count += 1
    print(f"[{message_count}] Time: {teensy_time}ms, Pos: {position:.3f}m, Accln: {acceleration:.2f}m/s2, Press: {pressure:.0f}Pa")
   
    # stop the control event once, control duration is reached
    if time.time() - main_timer > control_duration:
        break
    
np.save('MPC_log/log_file_test_1_tar2_2', log)
print(counter)

ser.write('<1:1000>'.encode())  # reduce thrust to decrease altitude
time.sleep(1)

ser.write('<1:0>'.encode())  # shut off valve
time.sleep(0.1)
ser.write('<1:0>'.encode())  # shut off valve
time.sleep(0.1)
ser.write('<1:0>'.encode())  # shut off valve

# close serial connection
ser.close()