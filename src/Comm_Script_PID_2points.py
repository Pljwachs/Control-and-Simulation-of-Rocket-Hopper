# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 14:36:14 2026

@author: felix
"""

# importing libraries
import serial  # Library for serial communication with python - install command: pip install pyserial
import time  # time tracking library
import numpy as np

# teensy loop time
loop_cycle_time = 1/60 # 60 measurement aquistions per second; can be adjusted to max of 66Hz
control_duration = 20 # in [s] --> your anticipated flight duration / control event duration

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

h = 1 / 60
trajectory = [2,1]
x_target = trajectory[0]
integral = 0
lasterr = 0
prev_position = 0
eqbm_count = 0
eqbm = 0
message_count = 0

## Nozzle
Th_d=9e-3                       # throat diameter [m]
Th_exit_d=11e-3                 # exit diameter [m]
A_throat=np.pi*(Th_d/2)**2      # area throat[m^2]
A_e=np.pi*(Th_exit_d/2)**2      # area exit nozzle[m^2]
epsilon=A_e/A_throat            # expansion ratio
T_4=273                         # [k]

## constant
R=8314             # universal gas constant [J/kmol*K]
M=28.013           # nitrogen gas  [kg/kmol]
gamma=1.4          # ratio of specific heats (diatomic gas)
g=9.81             # [m/s^2]

## mass
m_hop=3.5       # mass of hopper [kg]
m_hose=1.0      # mass of hose   [kg]
k_hose=6.0        # variable hopper mass [N/m]
F_RR=10         # rolling resistence [N]

# PID for pressure - Initial
kp = 4.5
ki = 0.85
kd = 4.0

# FF_Term
md4n_term=A_throat/np.sqrt(T_4*R/(M*gamma)) * np.power((gamma+1)/2, -(gamma+1)/(2*(gamma-1)))
ve_term=(2*gamma*R*T_4)/(gamma-1)/M

ve_suggest_val=float(np.sqrt(ve_term * (1 - np.power(0.1615, (gamma - 1) / gamma))))

thrust_req= m_hop*(g)+k_hose*x_target
pressure_req=thrust_req/(ve_suggest_val*md4n_term)
FF_term = 0.7*pressure_req/1e5

def pressure_PID(x_target, x, h, kp, ki, kd,FF_term):
    global integral, lasterr

    err = x - x_target
    integral = integral + ki * err * h
    derr = (err - lasterr) / h

    PIDlaw = -(kp * err + kd * derr + ki * integral) + FF_term

    if PIDlaw < 1:
        output = 1
    elif PIDlaw > 11:
        output = 11
    else:
        output = PIDlaw
    lasterr = err

    return output

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

    action = pressure_PID(x_target, position, h, kp, ki, kd,FF_term)
    action= int((action-1)/10*4095)
    log[:5, counter] = [teensy_time, acceleration, position, pressure, action]

    err_val = x_target - round(position,3)

    if err_val<0.2 and err_val>-0.2:
        eqbm_count = eqbm_count + 1
        # print(f"Equilibrium count: {eqbm_count}")
        # print(eqbm)
    else:
        eqbm_count = 0


    if eqbm_count > 50 and eqbm < len(trajectory)-1:
        eqbm = eqbm+1
        eqbm_count = 0
        x_target = trajectory[eqbm]
        print(f"New target position: {x_target}m")

    if eqbm == len(trajectory)-1 and eqbm_count > 50:
        control_duration = 0
     
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
    
np.save('PID_log/log_file_2points_2_2', log)
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