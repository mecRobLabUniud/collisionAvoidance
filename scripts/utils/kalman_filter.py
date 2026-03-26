#!/usr/bin/env python3
 
"""
░█░█░█▀█░█░░░█▄█░█▀█░█▀█░░░█▀▀░▀█▀░█░░░▀█▀░█▀▀░█▀▄
░█▀▄░█▀█░█░░░█░█░█▀█░█░█░░░█▀▀░░█░░█░░░░█░░█▀▀░█▀▄
░▀░▀░▀░▀░▀▀▀░▀░▀░▀░▀░▀░▀░░░▀░░░▀▀▀░▀▀▀░░▀░░▀▀▀░▀░▀


"""

import time
import numpy as np
import cupy as cp
from std_srvs.srv import Trigger  # Standard Reset Service
from ssm_definations.msg import HumanIK

from std_srvs.srv import Trigger  # Standard Reset Service
from ssm_definations.msg import HumanPositions, Humanfiltered

from math import cos,sin,tan,pi



class KalmanFilter():
    def __init__(self, shared_data1, shared_data2):
        # Filtering variables
        self.n = 65  # State vector size
        self.m = 46  # Measurement vector size
        self.p_n = 14  # Position states
        self.pl_n = 9   # Link lengths
        self.dt = 0.01  # Time step
        self.shared_data_camera1 = shared_data1
        self.shared_data_camera2 = shared_data2
        self.rate=0.01

        # message parameters
        self.prev_positions = None
        self.prev_time=None
        self.visibilityi1=[]
        self.visibilityi2=[]

        # Noise parameters
        self.var_p = 0.05
        self.var_pd = 0.05
        self.var_pdd = 0.05
        self.var_pddd = 0.05
        self.psi_z = 10
        self.prev_t1, self.prev_t2 = 0, 0
        self.prev_zk=np.zeros(46)
        
        # State vectors
        self.s_k = np.zeros(self.n)  # State estimate
        self.s_k[0]=np.array([3.0])
        self.s_k[56:]=np.array([ 0.535, 0.21, 0.355, 0.305, 0.535, 0.21, 0.355, 0.305, 0.15])

        self.p_k = np.eye(self.n)    # Covariance matrix

        # Process noise covariance
        self.Q = np.diag(np.hstack([
            np.ones(self.p_n) * (self.var_p ** 2) * (self.dt ** 4) / 24,
            np.ones(self.p_n) * (self.var_pd ** 2) * (self.dt ** 3) / 6,
            np.ones(self.p_n) * (self.var_pdd ** 2) * (self.dt ** 2) / 2,
            np.ones(self.p_n) * (self.var_pddd ** 2) * self.dt,
            np.zeros(self.pl_n)
        ]))

        # Measurement noise covariance
        self.R = self.psi_z * np.eye(self.m)

        self.I_=np.eye(self.p_n)                    # identity of length position states
        self.z1_=np.zeros([self.p_n,self.p_n])      # Zeros of length position states
        
        # State transition matrix (F_k)
        self.F_k = np.eye(self.n)  # Placeholder for state transition matrix
         # state transition matrix
        F1=np.hstack([self.I_, self.dt*self.I_, 0.5*(self.dt**2)*self.I_,
                      (self.dt**3)*self.I_/6]).reshape(self.p_n,self.p_n*4)
        F2=np.hstack([self.z1_, self.I_,  self.dt*self.I_, 0.5*(self.dt**2)*self.I_]).reshape(self.p_n,self.p_n*4)
        F3=np.hstack([self.z1_, self.z1_,  self.I_, self.dt*self.I_]).reshape(self.p_n,self.p_n*4)
        F4=np.hstack([self.z1_, self.z1_, self.z1_, self.I_]).reshape(self.p_n,self.p_n*4)
        self.F_k[0:56,0:56]=np.vstack([F1,F2,F3,F4]).reshape(self.p_n*4,self.p_n*4)
    
        self.H_k=np.zeros([self.m,self.n])                                                  # observation matrix
        self.H_k[0:(self.p_n*2), 0:self.p_n] = np.vstack([self.I_, self.I_])
        self.H_k[(self.p_n*2):self.m, -9:self.n] = np.vstack([np.eye(self.pl_n),
                                                         np.eye(self.pl_n)])

        self.timer = self.create_timer(self.rate, self.filter_loop)
        self.exit_loop = False
        self.reset_requested = False
        self.reset_threshold=2.0
        self.last_meas_time=0.0
        self.start_timef=0.0
        # Create a service for resetting the filter
        self.srv = self.create_service(Trigger, 'reset_filter', self.reset_callback)

    # def destroy_node(self):
    #     """Ensures graceful shutdown."""
    #     self.exit_loop = True
    #     #self.filter_thread.join()
    #     super().destroy_node()
    
    # def reset_callback(self, request, response):
    #     self.get_logger().info("Reset requested.")
    #     self.reset_requested = True  # Set flag to indicate reset
    #     self.filter_reset()  # Perform reset
    #     self.reset_requested = False  # Resume filter execution
    #     response.success = True
    #     response.message = "Filter reset successfully."
    #     return response

    def predict(self):
        """Kalman filter prediction step."""
        self.s_k = self.F_k.dot(self.s_k)  # State prediction
        self.p_k = self.F_k.dot(self.p_k).dot(self.F_k.T) + self.Q  # Covariance prediction

    def build_measurement_vector(self):
        """Builds the measurement vector z_k from shared buffers."""
        data1 = self.shared_data_camera1.read()
        data2 = self.shared_data_camera2.read() 
        z_k= np.zeros(46)
        try:
            #retreive IK for dectection from camera 1
            if self.shared_data_camera1.check_val :                    
                    if self.prev_t1<data1['time'] or not self.shared_data_camera1.new_data_ready :                          
                            self.prev_t1 =  data1['time']
                            self.no_new1=False
                            self.R[0:self.p_n,0:self.p_n]=  np.diag(data1['R_z'])                                                    
                            self.R[28:37,28:37]=np.diag(data1['R_l']) 
                            self.visibilityi1=data1['visb']                      
                    else:
                            self.R[0:self.p_n,0:self.p_n]= self.R[0:self.p_n,0:self.p_n] *2                                                    
                            self.R[28:37,28:37]=self.R[28:37,28:37] *2
                            self.no_new1=True
            else:                            
                self.no_new1=True                               

            #retreive IK for dectection from camera 2
            if self.shared_data_camera2.check_val:                    
                    if self.prev_t2<data2['time'] or not self.shared_data_camera2.new_data_ready :                                                                
                            self.prev_t2 =  data2['time']                                                    
                            self.no_new2=False
                            self.R[14:14+self.p_n,14:14+self.p_n] = np.diag(data2['R_z'])                                                       
                            self.R[-9:46,-9:46] = np.diag(data2['R_l'])   
                            self.visibilityi2=data2['visb']                                                                       
                    else:
                        self.R[14:14+self.p_n,14:14+self.p_n] =  self.R[14:14+self.p_n,14:14+self.p_n] *2      
                        self.R[-9:46,-9:46] = self.R[-9:46,-9:46] *2
                        self.no_new2=True 
            else:
                    self.no_new2=True

            z_k = np.vstack([
                np.array(data1['z']).reshape(14, 1), np.array(data2['z']).reshape(14, 1),
                np.array(data1['l']).reshape(9, 1), np.array(data2['l']).reshape(9, 1)
            ]).reshape(46)
            for i in range(len(z_k)):
                if np.isnan(z_k[i]):
                    if not np.isnan(self.prev_zk[i]):
                        z_k[i]=self.prev_zk[i]
                    else:
                        z_k[i]=0
                    self.R[i,i]=1e+6
            self.prev_zk=z_k
        except Exception as e:
             self.get_logger().error(f"Filter measurement build error: {e}")
        return z_k

    def update(self, z_k):
        """Kalman filter update step."""
        if z_k is None:
            return
        if  any([ not self.no_new2+self.no_new1,  not self.no_new2==self.no_new1]):

            y_k = z_k - self.H_k.dot(self.s_k)  # Measurement residual
            S = self.H_k.dot(self.p_k).dot(self.H_k.T) + self.R  # Measurement covariance
        
            #S_inv=np.linalg.inv(S)
            S_inv=cp.asnumpy(cp.linalg.inv(cp.array(S)))
            #print('S_INV', type(S_inv),S_inv.shape)
            
            #update prediction                  
            KG_k = self.p_k.dot(self.H_k.T).dot(S_inv)  # Kalman gain

            # Update state estimate
            self.s_k = self.s_k + KG_k.dot(y_k)
            self.p_k = (np.eye(self.n) - KG_k.dot(self.H_k)).dot(self.p_k)
            self.last_meas_time=time.time()  

        if time.time()-self.last_meas_time > self.reset_threshold:
                self.filter_reset()
                self.get_logger().warn(f"filter is reset")
                                    
    def compute_bounds(self,alpha,dt):
        """
            given the state vector and time_step the function calculates the position,
            velocity, and acceleration bounds.
            using the calculated bounds the state vector is saturated to the limits
        """

        plink_lim=[550,250,370,350,550,250,370,350,150] # [Rhip to Rshoulder_vertical,Rhip to Rshoulder_horz, Rshold to Relbow, Relbow to Rwrist, similarly left side , hip2neck(not used )]
        alpha_b=[[np.inf,-np.inf],[np.inf, -np.inf],[2.0, -2.0],[np.pi,-np.pi],
                [np.pi/2,-np.pi/6],[8*np.pi/9, -np.pi/20],[0,0],[0,0],[np.pi/2,-70*np.pi/180],
                [np.pi/2,-np.pi/6],[8*np.pi/9, -np.pi/20],[0,0],[0,0],[np.pi/2,-70*np.pi/180]]
        a_d_b=[[0.8, 0],[0.8, -0.8],[0.1,-0.1],[np.pi/4,-np.pi/4],
            [np.pi/10,-np.pi/10],[np.pi/2, -np.pi/2],[np.pi/2, -np.pi/2],[np.pi/2, -np.pi/2],[3*np.pi/10,-3*np.pi/10],
            [np.pi/10,-np.pi/10],[np.pi/2, -np.pi/2],[np.pi/2, -np.pi/2],[np.pi/2, -np.pi/2],[3*np.pi/10,-3*np.pi/10]]
        a_dd_b=[[0.1, -0.1],[0.1, -0.1],[0.01, -0.01],[np.pi/8,-np.pi/8],
                [np.pi/20,-np.pi/20],[np.pi/4, -np.pi/4],[np.pi/4, -np.pi/4],[np.pi/4, -np.pi/4],[3*np.pi/20,-3*np.pi/20],
                [np.pi/20,-np.pi/20],[np.pi/4, -np.pi/4],[np.pi/4, -np.pi/4],[np.pi/4, -np.pi/4],[3*np.pi/20,-3*np.pi/20]]
        
        # alpha -->> joint state vector[j1l,j2l,j3l,j4l,j1r,j2r,j3r,j4r]
        # update alpha bounds through inequality constraints 
        # for Alpha2_r and Alpha3_r for both right and left arms
        
        # left arm: Alpha2_l and Alpha3_l bounds
        a_1l=alpha[10]
        if a_1l>alpha_b[10][0] and a_1l>alpha_b[10][1]:
            a_1l=alpha_b[10][0]
        elif a_1l<alpha_b[10][0] and a_1l<alpha_b[10][1]:
            a_1l=alpha_b[10][1]
        
        a_2l=alpha[11]
    
        #print(a_1l,a_2l)
        # left arm:  a2
        a_2inf_l=np.deg2rad(-43) + a_1l/3
        a_2sup_l=np.deg2rad(153) - a_1l/6

        alpha_b[11]=[a_2sup_l,a_2inf_l]

        if a_2l>alpha_b[11][0] and a_2l>alpha_b[11][1]:
            a_2l=alpha_b[11][0]
        elif a_2l<alpha_b[11][0] and a_2l<alpha_b[11][1]:
            a_2l=alpha_b[11][1]

        #print(a_2inf_l,a_2sup_l)
        # left arm:  a3
        a_3inf_l=-np.pi/2 + 7*a_1l/9 - a_2l/9 + 2*a_1l*a_2l/810 
        a_3sup_l=np.deg2rad(60) + 4*a_1l/9 - 5*a_2l/9 + 5*a_1l*a_2l/810
        alpha_b[12]=[a_3sup_l,a_3inf_l]
        
        # right arm: Alpha2_r and Alpha3_r bounds
        a_1r=alpha[5]
        if a_1r>alpha_b[5][0] and a_1r>alpha_b[5][1]:
            a_1r=alpha_b[5][0]
        elif a_1r<alpha_b[5][0] and a_1r<alpha_b[5][1]:
            a_1r=alpha_b[5][1]

        a_2r=alpha[6]
        
        # right arm:  a2
        a_2inf_r=np.deg2rad(-43) + a_1r/3
        a_2sup_r=np.deg2rad(153) - a_1r/6
        alpha_b[6]=[a_2sup_r,a_2inf_r]
        
        if a_2r>alpha_b[6][0] and a_2r>alpha_b[6][1]:
            a_2r=alpha_b[6][0]
        elif a_2r<alpha_b[6][0] and a_2r<alpha_b[6][1]:
            a_2r=alpha_b[6][1]

        # right arm:  a3
        a_3inf_r=-np.pi/2 + 7*a_1r/9 - a_2r/9 + 2*a_1r*a_2r/810 
        a_3sup_r=np.deg2rad(60) + 4*a_1r/9 - 5*a_2r/9 + 5*a_1r*a_2r/810    
        alpha_b[7]=[a_3sup_r,a_3inf_r]
        s_kb=alpha
        for i in range(14):
            s_kb[i]=max(alpha_b[i][1],min(alpha[i],alpha_b[i][0]))

            ald_sup=min(a_d_b[i][0],(alpha_b[i][0]-alpha[i])/dt)
            ald_inf=max(a_d_b[i][1],(alpha_b[i][1]-alpha[i])/dt)
            s_kb[i+14]=max(ald_inf,min(alpha[i+14],ald_sup))

            aldd_sup=min(a_dd_b[i][0],(a_d_b[i][0]-alpha[i+14])/dt)
            aldd_inf=max(a_dd_b[i][1],(a_d_b[i][1]-alpha[i+14])/dt)
            s_kb[i+28]=max(aldd_inf,min(alpha[i+28],aldd_sup))
        for j in range(9):
            jj=56+j
            if s_kb[jj]>plink_lim[j]/1000:
                s_kb[jj]=plink_lim[j]/1000

        return s_kb
        
    def filter_reset(self):
        self.get_logger().warn(f"Filter reset: No person detected for {self.reset_threshold} seconds")

        #  State vector and covariance
        self.s_k = np.zeros(self.n)  # State vector
        self.s_k[0]=np.array([3])
        self.s_k[56:]=np.array([ 0.535, 0.21, 0.355, 0.305, 0.535, 0.21, 0.355, 0.305, 0.15])

        self.p_k = np.eye(self.n)  # Covariance matrix
        
        # Process noise covariance
        self.Q = np.diag(np.hstack([
            np.ones(self.p_n) * (self.var_p ** 2) * (self.dt ** 4) / 24,
            np.ones(self.p_n) * (self.var_pd ** 2) * (self.dt ** 3) / 6,
            np.ones(self.p_n) * (self.var_pdd ** 2) * (self.dt ** 2) / 2,
            np.ones(self.p_n) * (self.var_pddd ** 2) * self.dt,
            np.zeros(self.pl_n)
        ]))    
        
        self.R = self.psi_z * np.eye(self.m)  # Initial measurement noise matrix

    def filter_loop(self):
        """Main filter execution loop."""
        if self.exit_loop or self.reset_requested:
            return
        self.start_timef=time.time()
        self.predict()  # Step 1: Predict state
        z_k = self.build_measurement_vector()  # Step 2: Build measurement vector
        self.update(z_k)  # Step 3: Update state
        self.compute_bounds(self.s_k,self.rate) # Step 4: saturate the states
        self.publish_filtered_data()  # Step 5: Publish filtered data

    def publish_filtered_data(self):
        """Publishes the filtered data."""
        indices = [18, 5, 6, 7, 8, 9, 10, 20, 20, 20]
        
        if self.exit_loop:
            return
        try:
            msg = Humanfiltered()
            if self.prev_positions is None:
                 velocities=np.zeros([10]).tolist()
                 joint_positions=self.Kinematic1.Human_UBForwardK(self.s_k[:14],self.s_k[56:-1])
            else:
                 dt=float(time.time()-self.prev_time)
                 joint_positions=self.Kinematic1.Human_UBForwardK(self.s_k[:14],self.s_k[56:-1])
                 velocities=[]
                 for j in indices:                
                    velocities.append(np.linalg.norm((self.prev_positions[j]-joint_positions[j])/dt))
            joint_positions=np.array(joint_positions).reshape([21,3])
            msg.head=joint_positions[18]
            msg.l_shoulder=joint_positions[5]
            msg.r_shoulder=joint_positions[6]
            msg.l_elbow=joint_positions[7]
            msg.r_elbow=joint_positions[8]
            msg.l_wrist=joint_positions[9]
            msg.r_wrist=joint_positions[10]
            msg.l_hip=joint_positions[20]
            msg.r_hip=joint_positions[20]
            msg.human=joint_positions[20]
            self.prev_positions=joint_positions
            self.prev_time=time.time()

            msg.timestamp = self.get_clock().now().to_msg()
            msg.velocities = velocities
            #msg.visibility = np.diag(self.p_k[4:14,4:14]).tolist()
            try:
                vis1 = np.array(self.visibilityi1).reshape(10,)
            except:
                  vis1 =np.zeros(10,)
            try:
                vis2 = np.array(self.visibilityi2).reshape(10,)
            except:
                  vis2 =np.zeros(10,)
            #vis2 = np.array(self.visibilityi2).reshape(10,)
            vis = np.column_stack((vis1, vis2))  # Ensures shape (10,2)
            msg.visibility = np.max(vis, axis=1).tolist()  # Max along each row
            self.publisher.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Filter publishing error: {e}")
           # self.exit_loop = True


