import numpy as np
import casadi as ca
import casadi.tools as ca_tools
import time


class MPCController:
    def __init__(
            self,
            x0: np.ndarray,
            u0: np.ndarray,
            z0: np.ndarray,
            Q: np.ndarray,
            R: np.ndarray,
            Qf: np.ndarray,
            xs: np.ndarray,
            freq: float,
            N: int,
            itermax: int     
        )->None:

        """
        Initialize the MPC controller 
        - x0: initial state
        - u0: initial control
        - z0: initial algebraic state
        - Q: State cost matrix
        - R: Control cost matrix
        - Qf: Terminal state cost matrix
        - freq: Control frequency
        - xs: Target state
        - N: Prediction horizon
        """
        self.x0=x0
        self.u0=u0
        self.z0=z0
        self.xs=xs 

        self.Q=Q
        self.R=R
        self.Qf=Qf

        self.N=N

        self.freq=freq
        self.dt=1.0/freq

        self.x_pred=None
        self.u_pred=None

        self.itermax=itermax

        # initial setup for the solver and function
        self.solver=None
        self.f_dynamics=None
        self.n_states=None
        self.n_controls=None
        self.n_alg_states=None

    @staticmethod
    def dm_to_array(dm):
        """Convert CasADi DM to numpy array"""

        return np.array(dm.full())
    
    def dynamics(self,f,current_x, current_z, current_u): #real dynamics

        """
        RK4 integration for dynamics propagation
        f: CasADi function that returns (ode, alg)
        """

        k_1,alg_1=f(current_x,current_z,current_u)
        k_2,_=f(current_x+self.dt/2*k_1,current_z,current_u)
        k_3,_=f(current_x+self.dt/2*k_2,current_z,current_u)
        k_4,_=f(current_x+self.dt/2*k_3,current_z,current_u)

        predicted_state=current_x+self.dt/6*(k_1+2*k_2+2 * k_3 + k_4)

        return predicted_state

    
    def setup(self):
            
            """Setup the MPC optimization problem"""  
            ##### define the state and control input
            x=ca.SX.sym('x')
            v=ca.SX.sym('v')
            states=ca.vertcat(x,v)
            n_states=states.numel()

            p3_u=ca.SX.sym('p3_u')
            control=p3_u
            n_controls=control.numel()

    
            # Actuator delay is assumed to be as part of noise
            md4n=ca.SX.sym('md4n')
            pe=ca.SX.sym('pe')
            ve=ca.SX.sym('ve')
            alg_states=ca.vertcat(md4n,pe,ve)
            n_alg_states=alg_states.numel()


            self.n_states=n_states
            self.n_controls=n_controls
            self.n_alg_states=n_alg_states
            
            md4n_eq=(6.36e-5)*p3_u/7.6075*125/216                              # m_choked
            # Nozzle exit
            pe_eq=p3_u/ca.power(1.67712, 3.5)                          #exit pressure
            ve_eq=15.2                                                          #exit velocity
            alg=ca.vertcat(md4n-md4n_eq,pe-pe_eq,ve-ve_eq)
            
            # rhs
            dotx=v
            dotv=md4n*ve/3.5-9.81-10/3.5*ca.sign(v)
            ode=ca.vertcat(dotx,dotv)
            
            #dae={'x':x_all,'z':z_all,'p':p_all,'ode':ode,'alg':alg}
            
            # nonlinear function
            f=ca.Function('f',[states,alg_states,control],[ode,alg],['states','alg_states','control'],['ode','alg'])
            self.f_dynamics=f

            ## for mpc
            # Decision variables for the entire horizon
            # X will be of size (n_states x (N+1)) and U of size (n_controls x N)
            X=ca.SX.sym('X',n_states,self.N+1)
            Z=ca.SX.sym('Z',n_alg_states,self.N+1)
            U=ca.SX.sym('U',n_controls,self.N)
            
            ####
            P=ca.SX.sym('P', n_states+n_states)  #include the initial (x_0,v_0) and reference state (x_f, v_f)
                 
            # cost function
            obj=0.0

            # constrain vector
            g=[]
            g.append(X[:,0]-P[:n_states])
        
            # create the g
            for i in range(self.N):
                current_state=X[:,i]
                current_alg=Z[:,i]
                current_cntl=U[:,i]

                obj+=(current_state-P[n_states:]).T @ self.Q @ (current_state-P[n_states:])+current_cntl.T @ self.R @ current_cntl
                
                k_1,alg_1=f(current_state,current_alg,current_cntl)
                k_2,_=f(current_state+self.dt/2*k_1,current_alg,current_cntl)
                k_3,_=f(current_state+self.dt/2*k_2,current_alg,current_cntl)
                k_4,_=f(current_state+self.dt/2*k_3,current_alg,current_cntl)

                predicted_state=current_state+self.dt/6*(k_1+2*k_2+2 * k_3 + k_4)

                next_state=X[:,i+1]
                g.append(next_state-predicted_state)
                g.append(alg_1)

            # Terminal error
            xserror=X[:,self.N]-P[n_states:]
            obj+=xserror.T @ self.Qf @ xserror

            # add final algebraic constrain
            _,alg_final=f(X[:,self.N], Z[:,self.N], U[:,self.N-1])
            g.append(alg_final)

            # Stack all decision variables
            opt_vars = ca.vertcat(
            ca.reshape(X, n_states * (self.N + 1), 1),
            ca.reshape(Z, n_alg_states * (self.N + 1), 1),
            ca.reshape(U, n_controls * self.N, 1)
            )

            # Stack all constraints
            g = ca.vertcat(*g)

            # simulation parameters
            nlp_prob={
                'f':obj,
                'x':opt_vars,
                'g':g,
                'p':P                 
            }
            
            opts_setting = {
                'ipopt.max_iter': 100,
                'ipopt.print_level': 0,
                'print_time': 0,
                'ipopt.acceptable_tol':1e-8,
                'ipopt.acceptable_obj_change_tol':1e-6 
                }
            
            self.solver=ca.nlpsol('solver','ipopt',nlp_prob,opts_setting)

            # inequality constrains
            x_min=0.0
            x_max=5.0
            v_min=-10.0
            v_max=10.0


         
            md4n_min=0
            md4n_max=ca.inf
            pe_min=0
            pe_max=ca.inf
            ve_min=0
            ve_max=ca.inf

            

            u_min=1.0   #bar
            u_max=11.0  #bar

            lbx=[]
            ubx=[]

            for i in range(self.N+1):
                lbx.append(x_min)
                lbx.append(v_min)
                ubx.append(x_max)
                ubx.append(v_max)
            # Algebraic state bounds (typically unbounded or with physical limits) 
            for i in range(self.N+1):
                lbx.append(md4n_min)
                lbx.append(pe_min)
                lbx.append(ve_min)
                ubx.append(md4n_max)
                ubx.append(pe_max)
                ubx.append(ve_max)

            # control state bounds (typically unbounded or with physical limits) 
            for i in range(self.N):
                 lbx.append(u_min)
                 ubx.append(u_max)


            self.args={
                'lbg': [0.0]*g.numel(),
                'ubg': [0.0]*g.numel(),
                'lbx': lbx,
                'ubx': ubx
            }

            return self.solver,f
    
    def compute_action(self,x0=None,z0=None,xs=None):
            # initialization
            """
            Run the MPC controller
        
            x0: Initial state (if None, use self.x0)
            xs: Reference state (if None, use self.xs)
        
            Returns: control output
            """

            if self.solver is None: 
                 print("the solver didn't call, please run the setup() beforehand")
            if x0 is None:
                 x0=self.x0
            if xs is None:
                 xs=self.xs
            if z0 is None:
                 z0=self.z0

            # initialize
            state_0=ca.DM(x0)
            state_ref=ca.DM(xs)

            # initial guess for optimization variables
            X0=ca.repmat(state_0, 1, self.N+1)
            Z0=ca.repmat(ca.DM(self.z0),1,self.N+1)
            u0_init=ca.DM(self.u0).reshape((self.n_controls,1))
            u0=ca.repmat(u0_init,1,self.N)
            
            # store for trajectory
            cat_states = self.dm_to_array(X0)
            cat_alg_states = self.dm_to_array(Z0)
            cat_controls =self.dm_to_array(u0[:, 0])
            times = np.array([[0]])

            mpc_iter=0
            
        
            while ca.norm_2(state_0-state_ref)>1e-2 and mpc_iter<self.itermax:
                t1=time.time()
                
                #set parameter
                p=ca.vertcat(state_0,state_ref)

                #set initial guess

                X0_v = X0.reshape((-1, 1))
                Z0_v = Z0.reshape((-1, 1))
                u0_v = u0.reshape((-1, 1))
                x0_opt = ca.vertcat(X0_v,Z0_v,u0_v)

                res=self.solver(x0=x0_opt,lbx=self.args['lbx'],ubx=self.args['ubx'],lbg=self.args['lbg'],ubg=self.args['ubg'],p=p)
    
    
                # Extract the control sequence from the solution. The state trajectory is first,
                # so the first control is located at index = n_states*(N+1)
                sol=res['x'].full().flatten()
                
                idx=0
                X_sol=sol[idx:idx+self.n_states*(self.N+1)].reshape(self.n_states,self.N+1,order='F')
                idx=self.n_states*(self.N+1)

                Z_sol=sol[idx:idx+self.n_alg_states*(self.N+1)].reshape(self.n_alg_states,self.N+1,order='F')
                idx+=self.n_alg_states*(self.N+1)

                U_sol=sol[idx:idx+self.n_controls*self.N].reshape(self.n_controls,self.N,order='F')

                #store trajectory
                cat_states = np.column_stack((cat_states, X_sol[:, 0]))
                cat_alg_states = np.column_stack((cat_alg_states, Z_sol[:, 0]))
                cat_controls = np.column_stack((cat_controls, U_sol[:, 0]))

                # Propagate the system using the discrete dynamics f (Euler integration)
                # Note: We use the same function f defined earlier with CasADi operators
                u_applied=ca.DM(U_sol[:,0])
                z_current=ca.DM(Z_sol[:,0])
                state_0= self.dynamics(self.f_dynamics,state_0, z_current,u_applied)

                t2=time.time() 
                
                times=np.vstack((times,t2-t1))
                
                mpc_iter+=1
                
                if mpc_iter % 10 == 0:
                    print(f'Iteration {mpc_iter}, error: {float(ca.norm_2(state_0 - state_ref)):.6f}')

            
            print(f'\nSteady state error: {float(ca.norm_2(state_0 - state_ref)):.6e}')
            print(f'Total iterations: {mpc_iter}')
            print(f'Average solve time: {np.mean(times):.4f} s')

            return float(cat_controls[:,0].squeeze())


        



