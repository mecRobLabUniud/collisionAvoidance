import threading
import time
import numpy as np
import cupy as cp
import rclpy
from rclpy.node import Node
from rclpy.service import Service
from std_srvs.srv import Trigger  # Standard Reset Service
from ssm_definations.msg import HumanIK

from std_srvs.srv import Trigger  # Standard Reset Service
from ssm_definations.msg import HumanPositions, Humanfiltered

from math import cos,sin,tan,pi

class Kinematic:
    def __init__(self):
         self.data=None

    def Human_UBInverseK(self,right,left,hip):
        # % * _ |ARM Inverse Kinemantics| _ * 
        # % ........................... get hip angle................................
                        
        hipr=hip[0,:]
        hipl=hip[1,:]
        hip_pos=0.5*(hipl+hipr)
        #print('hip_pos',hip_pos)
        hipvec=(hipl-hipr)
        hip_proj=[np.dot(hipvec,[1,0,0]),np.dot(hipvec,[0,1,0]),0]
        hip_zvrs= hip_proj/np.linalg.norm(hip_proj)
        hip_yvrs= np.cross(hip_zvrs, [0,0,1])


        T_hW=np.eye(4)
        T_hW[0:3,0]=[0,0,1]
        T_hW[0:3,1]=hip_yvrs
        T_hW[0:3,2]=hip_zvrs
        T_hW[0:3,3]= hip_pos

        P_shr= np.dot(np.linalg.inv(T_hW),np.hstack([right[0,:],np.array(1)]).T)

        tau_r=np.arctan2(P_shr[1],P_shr[0])
        # %rad2deg(tau)

        # % ...............Get shoulder angles alpha1 alpha2.........................

        # % shoulder angles are obtained through the orientation 'Z-frame' of the 3rd
        # % joint (calculated as the unit vector from shoulder to elbow)

        P_ehr= np.dot(np.linalg.inv(T_hW),np.hstack([right[1,:],np.array(1)]).T)
        z_3r=(np.array(P_ehr[0:3]-P_shr[0:3])/np.linalg.norm(P_shr[0:3]-P_ehr[0:3]))


        # % equations for reference
        # % c1.c2 =-s2.Tt -z_X/ct
        # % s2=(z_y - z_x.Tt)/(ct + st.Tt)
        # % T1= (-Z_z/c2)/(c1.c2/c2)
        Alpha2_r=np.arcsin((z_3r[1]-z_3r[0]*tan(tau_r))/(cos(tau_r)+(tan(tau_r)*sin(tau_r))))                                        #% joint 2
        Alpha1_r=np.arctan2(-z_3r[2]/np.sign(cos(Alpha2_r)), ((-sin(Alpha2_r)*tan(tau_r))- (z_3r[0]/cos(tau_r)))/np.sign(cos(Alpha2_r)))  # % joint 1

        
        
        # % .................get alpha3 and elbow angles.............................
        """ 
            % the joint 4 located at elbow is obtained as the angle between the 
            unit vector from shoulder to elbow(Z_3) and  unit vector from elbow to
            % wrist(x_4) in the hip reference frame

            x_4 (calculated as the unit vector from elbow to wrist  position in Hip reference frame)
        """
        P_whr=np.dot(np.linalg.inv(T_hW),np.hstack([right[2,:],np.array(1)]).T)
        x_4r=(np.array(P_ehr[0:3]-P_whr[0:3])/np.linalg.norm(P_ehr[0:3]-P_whr[0:3]))

        # % alpha4= acos(dot(z_3,x_4)) + pi/2 (home angle wrt z_3
        pl_r=np.dot(z_3r,x_4r)
        Alpha4_r=np.arccos(pl_r) - pi/2

        T_r4=[[-cos(Alpha1_r)*sin(Alpha2_r)*cos(tau_r) + cos(Alpha2_r)*sin(tau_r), -sin(Alpha1_r)*cos(tau_r), - sin(Alpha2_r)*sin(tau_r) - cos(Alpha1_r)*cos(Alpha2_r)*cos(tau_r)],
            [-cos(Alpha2_r)*cos(tau_r) - cos(Alpha1_r)*sin(Alpha2_r)*sin(tau_r), -sin(Alpha1_r)*sin(tau_r),   sin(Alpha2_r)*cos(tau_r) - cos(Alpha1_r)*cos(Alpha2_r)*sin(tau_r)],
            [                                  -sin(Alpha1_r)*sin(Alpha2_r),         cos(Alpha1_r),                                    -cos(Alpha2_r)*sin(Alpha1_r)]]

        # % the alpha 3 is caluclated as the by comparing the R_wh= R_3h* R_w3 =>
        # % inv(R_3h)*R_wh(1:3,1) = R_w3(1:3,1)
        T_hs3=np.array(T_r4).reshape(3,3)
        x_34r=np.dot(np.linalg.inv(T_hs3),x_4r)
        Alpha3_r=np.arctan2(x_34r[1]/np.sign(cos(Alpha4_r)),x_34r[0]/np.sign(cos(Alpha4_r)))

        # %
        # % .........................get hip angle left..............................

        P_sh_l= np.dot(np.linalg.inv(T_hW),np.hstack([left[0,:],np.array(1)]).T)
        tau_l=np.arctan2(P_sh_l[1],P_sh_l[0])

        # % ..................Get shoulder angles alpha1l alpha2l....................
        """ 
            shoulder angles are obtained through the orientation 'Z-frame' of the 3rd
            % joint (calculated as the unit vector from shoulder to elbow)
        """
        P_eh_l= np.dot(np.linalg.inv(T_hW),np.hstack([left[1,:],np.array(1)]).T)
        z_3l=((P_sh_l[0:3]-P_eh_l[0:3])/np.linalg.norm(P_sh_l[0:3]-P_eh_l[0:3]))

        Alpha2_l=np.arcsin((-z_3l[1]+z_3l[0]*tan(tau_l))/(cos(tau_l)+tan(tau_l)*sin(tau_l)))                                 # % joint 2
        Alpha1_l=np.arctan2(-z_3l[2]/np.sign(cos(Alpha2_l)),(-sin(Alpha2_l)*sin(tau_l) + z_3l[0])/(cos(tau_l)*np.sign(cos(Alpha2_l)))) #% joint 1

       
        # % ......................get alpha3 and elbow angles........................

        # % the joint 4 located at elbow is obtained as the angle between the 
        # % unit vector from shoulder to elbow(Z_3l) and  unit vector from elbow to
        # % wrist(x_4l) in the hip reference frame


        # % x_4l (calculated as the unit vector from elbow to wrist position 
        # % in Hip reference frame
        P_whl= np.dot(np.linalg.inv(T_hW),np.hstack([left[2,:],np.array(1)]).T)
        x_4l=((P_eh_l[0:3]-P_whl[0:3])/np.linalg.norm(P_eh_l[0:3]-P_whl[0:3]))

        # % alpha4= acos(dot(z_3,x_4)) + pi/2 (home angle wrt z_3)
        pl_l=-np.dot(z_3l,x_4l)
        Alpha4_l=np.arccos(pl_l) - pi/2

        T_l4=[[-cos(Alpha1_l)*sin(Alpha2_l)*cos(tau_l) + cos(Alpha2_l)*sin(tau_l),  -sin(Alpha1_l)*cos(tau_l), sin(Alpha2_l)*sin(tau_l) + cos(Alpha1_l)*cos(Alpha2_l)*cos(tau_l)],
            [-cos(Alpha2_l)*cos(tau_l) - cos(Alpha1_l)*sin(Alpha2_l)*sin(tau_l),    -sin(Alpha1_l)*sin(tau_l),  -sin(Alpha2_l)*cos(tau_l) + cos(Alpha1_l)* cos(Alpha2_l)*sin(tau_l)],
            [                                 sin(Alpha2_l)*sin(Alpha2_l),               -cos(Alpha1_l),            -cos(Alpha2_l)*sin(Alpha1_l)]]

    
        T_hs3l=np.array(T_l4).reshape(3,3)
        x_34l=np.dot(np.linalg.inv(T_hs3l),x_4l)
        Alpha3_l=np.arctan2(x_34l[1]/np.sign(np.cos(Alpha4_l)),x_34l[0]/np.sign(np.cos(Alpha4_l)))
       
        # % heading angle    
        heading_ang=np.arctan2(hip_zvrs[0],-hip_zvrs[1])
    
        z_k=[hip_pos[0],hip_pos[1],hip_pos[2],heading_ang,tau_r,Alpha1_r,Alpha2_r,Alpha3_r,Alpha4_r,tau_l,Alpha1_l,Alpha2_l,Alpha3_l,Alpha4_l]
       
        vec1r=right[0,:]-hip_pos
        l1_r=np.dot(vec1r,[0,0,1])  ## world frame torso height
        l2_r=np.linalg.norm([np.dot(vec1r,[1,0,0]),np.dot(vec1r,[0,1,0]),0])  ## world frame shoulder half width
        l3_r=np.linalg.norm(right[0,:]-right[1,:])
        l4_r=np.linalg.norm(right[2,:]-right[1,:])
        vec1l=left[0,:]-hip_pos
        l1_l=np.dot(vec1l,[0,0,1])   ## world frame torso height
        l2_l=np.linalg.norm([np.dot(vec1l,[1,0,0]),np.dot(vec1l,[0,1,0]),0])  ## world frame shoulder half width
        l3_l=np.linalg.norm(left[0,:]-left[1,:])
        l4_l=np.linalg.norm(left[2,:]-left[1,:])
        l_k=[l1_r,l2_r,l3_r,l4_r,l1_l,l2_l,l3_l,l4_l]
       
        return z_k,l_k

    def dh_calc(self,a, alpha, d, theta ):
            # %DH Summary of this function a, alpha, d, theta  are inputs
            # % 
            # %   Detailed explanation goes here

            T =np.array([[cos(theta), -sin(theta) * cos(alpha),  sin(theta) * sin(alpha), a*cos(theta)],
                [sin(theta),   cos(theta) * cos(alpha), -cos(theta) * sin(alpha), a*sin(theta)],
                [0.0,          sin(alpha),               cos(alpha),              d],
                [0.0,          0.0,                      0.0,                     1]]).reshape(4,4)
            
            # % T = [cos(theta),                -sin(theta),                  0 ,                 a;
            # %     sin(theta)*cos(alpha),   cos(theta)*cos(alpha),   -sin(alpha),    -sin(alpha)*d;
            # %     sin(theta)*sin(alpha),   cos(theta)*sin(alpha),   cos(alpha),     cos(alpha)*d;
            # %     0.0,          0.0,                      0.0,                     1];

            return T
        
    def  Human_UBForwardK(self,z_k,l_k):
        x_0=z_k[0]
        y_0=z_k[1]
        z_0=z_k[2]
        theta_h=z_k[3]
        tau_r=z_k[4]
        Alpha1_r=z_k[5]
        Alpha2_r=z_k[6]
        Alpha3_r=z_k[7]
        Alpha4_r=z_k[8]
        tau_l=z_k[9]
        Alpha1_l=z_k[10]
        Alpha2_l=z_k[11]
        Alpha3_l=z_k[12]
        Alpha4_l=z_k[13]
        
        l1_r=l_k[0]
        l2_r=l_k[1]
        l3_r=l_k[2]
        l4_r=l_k[3]
        l1_l=l_k[4]
        l2_l=l_k[5]
        l3_l=l_k[6]
        l4_l=l_k[7]

        # % define dh matrix of hip_heading to hip tilt 
        dh_w2h=[[0, pi/2, 0, 0],
                [0,   0,    0, pi/2],
                [0, theta_h, 0, 0]]

        # % define dh matrix of hip tilt to right arm kinematic chain
        # % note the joint 4 defined is fixed joint 
        dhparams_r=[[l1_r, pi/2, -l2_r, tau_r],
                [0, pi/2, 0, Alpha1_r],
                [0,    0, 0,pi/2+Alpha2_r],
                [0, -pi/2, 0,  0],
                [0, -pi/2, l3_r, Alpha3_r],
                [-l4_r, 0, 0,  Alpha4_r]]

        # % define dh matrix of hip tilt to left arm kinematic chain
        # %note the joint 4 defined is fixed joint 
        dhparams_l=[[l1_l, -pi/2, l2_l, tau_l],
                [0, -pi/2, 0, Alpha1_l],
                [0, 0, 0,pi/2+Alpha2_l],
                [0, pi/2, 0, 0],
                [0, pi/2, -l3_l, Alpha3_l],
                [-l4_l, 0, 0,  Alpha4_l]] 

        T0=np.eye(4)
        T0[0:3,3]=[x_0,y_0,z_0]
        dh_wl=len(dh_w2h)
        k_w=[] 
        T_w=[]# % world to current joint transformation
        t_w=T0
        for i in range(dh_wl):
            k_w.append(self.dh_calc(dh_w2h[i][0],dh_w2h[i][1],dh_w2h[i][2],dh_w2h[i][3]))
            t_w=np.matmul(t_w,k_w[i])
            T_w.append(t_w)
        
        # % obtain forward kinematics of  hip tilt to right arm 
        dh_l=len(dhparams_r)
        k_r=[]; #% parent to child transformation
        T_r=[];# % world to current joint transformation
        t_r= np.eye(4)
        for i in range(dh_l):
            k_r.append(self.dh_calc(dhparams_r[i][0],dhparams_r[i][1],dhparams_r[i][2],dhparams_r[i][3]))
            t_r=np.matmul(t_r,k_r[i])
            T_r.append(t_r)

        #% obtain forward kinematics of  hip tilt to left arm 
        k_l=[]#% parent to child transformation
        T_l=[]#; % world to current joint transformation
        t_l=np.eye(4)
        for i in range(dh_l):
            k_l.append(self.dh_calc(dhparams_l[i][0],dhparams_l[i][1],dhparams_l[i][2],dhparams_l[i][3]))
            t_l=np.matmul( t_l,k_l[i])        
            T_l.append(t_l)
    
        right_=np.array([T_w[2].dot(T_r[0][0:4,3]), T_w[2].dot(T_r[4][0:4,3]), T_w[2].dot(T_r[5][0:4,3])]).reshape(3,4)
        left_=np.array([T_w[2].dot(T_l[0][0:4,3]), T_w[2].dot(T_l[4][0:4,3]), T_w[2].dot(T_l[5][0:4,3])]).reshape(3,4)
        hip_=np.array([T_w[1].dot([0,0,-l2_r,1]),T_w[1].dot([0,0,l2_l,1])]).reshape(2,4)
        right=right_[:,0:3]
        left=left_[:,0:3]
        hip=hip_[:,0:3]
        # T_rn=zeros(4,4,dh_l);
        # T_ln=zeros(4,4,dh_l);
        # for i =1:dh_l
        #     T_rn(:,:,i)=eval(T_w(:,:,2))*eval(T_r(:,:,i));
        #     T_ln(:,:,i)=eval(T_w(:,:,2))*eval(T_l(:,:,i));
        # end
        x_neck=(T_w[2].dot(T_r[0]))
        #print(x_neck[0:3,0]*0.15)
        kps=np.zeros([21,3])
        kps[:,:]=np.nan
        kps[5,:]=left[0,:]
        kps[7,:]=left[1,:]
        kps[9,:]=left[2,:]
        kps[6,:]=right[0,:]
        kps[8,:]=right[1,:]
        kps[10,:]=right[2,:]
        kps[20,:]=[x_0,y_0,z_0]
        kps[19,:]=(left[0,:] +right[0,:])*0.5
        kps[18,:]=(left[0,:] +right[0,:])*0.5 + x_neck[0:3,0]*0.15
        kps=kps.reshape([21,3])
        return kps
  



class FilterNode(Node):
    def __init__(self,shared_data1,shared_data2):
        super().__init__('filter')

        # ROS2 Publisher
        self.publisher = self.create_publisher(Humanfiltered, '/filter_out', 10)  
        

        # Filtering variables
        self.n = 65  # State vector size
        self.m = 46  # Measurement vector size
        self.p_n = 14  # Position states
        self.pl_n = 9   # Link lengths
        self.dt = 0.01  # Time step
        self.shared_data_camera1 = shared_data1
        self.shared_data_camera2 = shared_data2
        self.Kinematic1=Kinematic()
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


    def destroy_node(self):
        """Ensures graceful shutdown."""
        self.exit_loop = True
        #self.filter_thread.join()
        super().destroy_node()
    
    def reset_callback(self, request, response):
        self.get_logger().info("Reset requested.")
        self.reset_requested = True  # Set flag to indicate reset
        self.filter_reset()  # Perform reset
        self.reset_requested = False  # Resume filter execution
        response.success = True
        response.message = "Filter reset successfully."
        return response

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


