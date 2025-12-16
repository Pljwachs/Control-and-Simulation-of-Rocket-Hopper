# Required Packages
import numpy as np
import matplotlib.pyplot as plt


###########################################################

# Parameter Setup
# Integration parameter 
h=1/5000  # step size    need at least 1000hz (sec)
t0=0    # initial time (sec)
tf=10    # final time (sec)

time=np.linspace(t0,tf,int((tf-t0)/h)+1)

## States
#1. Hose / 2. Fitting /3. Main Valve/4. Nozzle
# Initial pressure (Pa)
p_s=11e5            #p_s pressure supply: at the inlet of the hose   
p1_t0=p_s           # inlet of the fitting
p2_t0=p_s           # inlet of the valve
p3_t0=1e5           # inlet of the nozzle:  assume initial constant air pressure

# Initial mass flow rate (kg/s)
md1_t0=0
md2_t0=0
md3_t0=0
md4_t0=0

state_t0=np.array([p1_t0,md1_t0,p2_t0,md2_t0,p3_t0,md3_t0,md4_t0])
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

x_target=3.0         # height target [m]

motion_t0=np.array([x_t0,v_t0])
motion_t=np.zeros((len(motion_t0),len(time)))

## PID parameters (gain has to be larger since the step size is really small)
kp=4.0
ki=0.75
kd=2.0

integral=0   # initialize the integral
lasterr=0    # for the kd  



###########################################################

# System & environmental parameters

## Hose
H_l=8                  # length [m]
H_d=19.3e-3            # inner diameter[m]
L_1=8                  #  [m]
A_1=np.pi*(H_d/2)**2   # [m^2]
V_1=L_1*A_1                # [m^3] A_1*H_l  


## Fitting
F_l=49e-3                      # length [m]
F_d=11.9e-3                    # diameter [m]
A_2=np.pi*(F_d/2)**2           #[m^2]
V_2=A_2*F_l                    #[m^3] #A_2*F_l


## Main Valve
Kv=4.8 
Open_t=825e-3    # opening time [s]
Close_t=1700e-3  # closing time[s]
T_3=273          # [K]


## Nozzle
Th_d=9e-3                        # throat diameter [m]
Th_exit_d=11e-3                  # exit diameter [m]
A_throat=np.pi*(Th_d/2)**2      #[m^2]
A_e=np.pi*(Th_exit_d/2)**2      #[m^2]
epsilon=A_e/A_throat           #expansion ratio
T_4=273                        #[k]


###########################################################

## mass
m_hop=3.5    # mass of hopper [kg]
m_hose=1     # mass of hose   [kg]
prev_pose=x_t0  # for hose dynamic modeling
k_hose=6     # variable hopper mass [N/m]
F_RR=10      # rolling resistence [N]

## constant
R=8314            # universal gas constant [J/kmol*K]
M=28.013           # nitrogen gas  [kg/kmol]
gamma=1.4          # ratio of specific heats (diatomic gas)
g=9.81             # [m/s^2]


f=0.0072                   #  f=0.316/Re^0.25  # darcy friction factor, assuming nitrogen gas flow rate of 1kg/s 
# Re=4*dotm/(\mu * pi* D(diameter))  dynamic viscosity is assume to be 1.76x10^-5 [Pa*s] 20 degree C and 1 atm 
lam_1=f*L_1/H_d             #lam_1=f*L_1/H_d  #for hose
lam_2=lam_1*100                 # for fitting
###########################################################
## to do list: correct the lam_1 lam_2 f values 


print(lam_1)



def Kvconvert(rho,T,K_v,p_1,p_2):
    # in here p1 p2 in bar (/1e5)
    p_1=p_1/1e5
    p_2=p_2/1e5

    if p_2!=p_1:
      if p_2/p_1>=0.5:
         Q_N=K_v*514/np.sqrt(rho*T/(p_1-p_2)/p_2)    #Nm^3/h normal cubic meters per hour
         dotm=Q_N*rho/36
      else :
         Q_N=K_v*(257*p_1)/np.sqrt(rho*T)
         dotm=Q_N*rho/36
    else:
       dotm=0                            
    
    return dotm



# Butcher Array
RK4matrix=np.array([
      [0,0,0,0],
      [1/4,1/4,0,0],
      [27/40,-189/800,729/800,0],       
      [1,214/891,1/33,650/891],       
      [214/891,1/33,650/891,0],      
   ])


def rk4_ex(f,t,x,h,*args):  #*args allows unknown number of arguments
   k1=f(t+h*RK4matrix[0][0],x+h*RK4matrix[0][1],*args)
   k2=f(t+h*RK4matrix[1][0],x+h*k1*RK4matrix[1][1],*args)
   k3=f(t+h*RK4matrix[2][0],x+h*k2*RK4matrix[2][1],*args)
   k4=f(t+h*RK4matrix[3][0],x+h*k3*RK4matrix[3][1],*args)

   xnext=x+h*(k1*RK4matrix[4][0]+k2*RK4matrix[4][1]+k3*RK4matrix[4][2]+k4*RK4matrix[4][3])
   return xnext


###########################################################
## ODE functions

def capacitor(t,p0,K0,V0,rho_0,md_0,md_1):   #
   return K0/(V0*rho_0)*(md_0-md_1)                              # p_dot

                                               
def inductor(t,md0,p_0,p_1,lam,rho_0,A,L):     #md1,h,t,p1,p2,lam_1,rho_1,A_1,L_1
   return (p_0-p_1 -lam/(2*rho_0*A**2)*md0**2*np.sign(md0))*A/L  #md1 dot

def dx(t,motion,dotm,ve,*args):
   xdot=motion[1]
   xdotdot=dotm*ve/m_hop-g-F_RR/m_hop*np.sign(xdot)-k_hose/m_hose*xdot*h   # the last term (hose) assumes that the hose become the damping term
   return np.array([xdot,xdotdot])



# main update function

def stateupdate(t,h,x,rho,motion):
   global prev_pose

   def getK(p):
      return p*gamma
   
   def get_rho(p_t1,p_t2,rho_1):              # get density from adiabatic relation
      return rho_1*np.power(p_t2/p_t1,1/gamma)  # careful
  
   # states
   p1=x[0]
   md1=x[1]
   p2=x[2]
   md2=x[3]
   p3=x[4]
   md3=x[5]
   md4=x[6]


   # motions
   x=motion[0]
   v=motion[1]

   #get K
   K1=getK(p1)
   K2=getK(p2)

   #get density
   rho_1=rho[0]
   rho_2=rho[1]
   rho_3=rho[2]

   ###########################################################
   # Algebraic Equation  (note: equations with "dot" use RK4)
   
   # Hose p1
   p1n=rk4_ex(capacitor,t,p1,h,K1,V_1,rho_2,md1,md2)          # 
   p2n=p3_actual    
                                    

   # Fitting p2

   md2n=md3                                         # assume same mass flow rate
   p2n=p1-lam_2/(2*rho_2*A_2**2)*md2**2*np.sign(md2)

   # Main Valve 
   md3n=np.sqrt(np.abs(p2-p3)/(3600**2*1e5/rho_3**2/Kv**2))              #Kvconvert(rho_3,T_3,Kv,p2,p3)
                                 # capacitor property

   # Nozzle
   p3_u=#command input pressure                                      # control input  (introduce the time delay modify once imply the controller)

   # intorduce the actuator delay
   if p3_u>p3:                 # opening lag
      deltat=Open_t*(p3_u/1100000)
   elif p3_u<p3:               # closing lag
      deltat=Close_t*(1-p3_u/1100000)
   else:                       # same input (not closing/opening)
      deltat=h
   
   p3_actual=p3_u                         # discard the differential equation-> use time lag modelling
   md4n=A_throat*p3_actual/(np.sqrt(T_4*R/(M*gamma)))*np.power((gamma+1)/2,-(gamma+1)/(2*(gamma-1)))  # m_choked

   # Nozzle exit
   pe=p3_actual/np.power(1+((gamma-1)/2*Me**2), gamma/(gamma-1))                         #exit pressure
   ve=np.sqrt((2*gamma*R*T_4)/(gamma-1)/M*(1-np.power(pe/p3_actual,(gamma-1)/gamma)))    #exit velocity
   

   ###########################################################
   # ODE integrator with RK4 
   # update md1dot / p1_dot / p2_dot  through Runge-Kutta 4 method 
   
   md1n=rk4_ex(inductor,t,md1,h,p_s,p1,lam_1,rho_1,A_1,L_1)   # rk4_ex rules: t,x,h, *args
                           # due to agrressive transient response I couldn't use rk4_ex(capacitor,t,p2,h,K2,V_2,rho_2,md2,md3)
                                                  

   print("ve: ",ve,"pe: ",pe," p3: ",p3_actual)
   #print("thrust [N]", md4n*ve)
   # update the motion
   motion_n=rk4_ex(dx,t,motion,h,md4n,ve)
 
   ###########################################################

   rho_1n=rho_1                          # as the pressure at the supply assumed to be fixed, it won't change
   rho_2n=get_rho(p1,p1n,rho_2)
   rho_3n=get_rho(p2,p2n,rho_3)

   return [p1n,md1n,p2n,md2n,p3_actual,md3n,md4n],[rho_1n,rho_2n,rho_3n],motion_n



# main iteration loop

# initialize the state at each time stamp
state_ti=state_t0
rho_ti=rho_t0
motion_ti=motion_t0


for ti in range(len(time)-1):
    #
    t=time[ti]

    #store the step
    state_t[:,ti]=state_ti
    rho_t[:,ti]=rho_ti
    motion_t[:,ti]=motion_ti

    # calculte the new value (RK4/Algebraic update)
    state_ti,rho_ti,motion_ti=stateupdate(t,h,state_ti,rho_ti,motion_ti)

    prev_pose=motion_t[0,ti-1] if ti>0 else x_t0

