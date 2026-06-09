// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#include <cmath>
#include <iostream>
#include <fstream>

#include <franka/exception.h>
#include <franka/robot.h>

#include "examples_common.h"

/**
 * @example simple_control.cpp
 * An example showing how to generate a joint position motion.
 *
 * @warning Before executing this example, make sure there is enough space in front of the robot.
 */

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <robot-hostname>" << std::endl;
    return -1;
  }

  int columns = 7;
  bool stamp_results = false;

  try {
    franka::Robot robot(argv[1]);
    setDefaultBehavior(robot);

    // First move the robot to the starting waypoint
    std::array<double, 7> q;

    std::ifstream fileq("/home/panda/Desktop/TRAJ_EXP/Cpp/q.txt");
    if (fileq.is_open()) {
      for (int i=0; i<columns; i++) {
        fileq >> q[i];
      }  
    }

    std::array<double, 7> q_p;

    std::ifstream fileqp("/home/panda/Desktop/TRAJ_EXP/Cpp/q_p.txt");
    if (fileqp.is_open()) {
      for (int i=0; i<columns; i++) {
        fileqp >> q_p[i];
      }  
    }

    std::array<double, 7> tau;

    std::ifstream filetau("/home/panda/Desktop/TRAJ_EXP/Cpp/tau.txt");
    if (filetau.is_open()) {
      for (int i=0; i<columns; i++) {
        filetau >> tau[i];
      }  
    }

    float t;

    std::ifstream filet("/home/panda/Desktop/TRAJ_EXP/Cpp/t.txt");
    if (filet.is_open()) {
      filet >> t;
    }

    std::ofstream fileqexp("/home/panda/Desktop/TRAJ_EXP/Cpp/q_BASE_exp.txt");

    std::ofstream fileqpexp("/home/panda/Desktop/TRAJ_EXP/Cpp/q_p_BASE_exp.txt");

    std::ofstream filetauexp("/home/panda/Desktop/TRAJ_EXP/Cpp/tau_BASE_exp.txt");

    std::ofstream filetexp("/home/panda/Desktop/TRAJ_EXP/Cpp/t_BASE_exp.txt");  

    MotionGenerator motion_generator(0.5, q);
    std::cout << "WARNING: This example will move the robot! "
              << "Please make sure to have the user stop button at hand!" << std::endl
              << "Press Enter to continue..." << std::endl;
    std::cin.ignore();
    robot.control(motion_generator);
    std::cout << "Finished moving to initial joint configuration." << std::endl;

    // Set additional parameters always before the control loop, NEVER in the control loop!
    // Set collision behavior.
    robot.setCollisionBehavior(
        {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
        {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}}, {{20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0}},
        {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}},
        {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}}, {{20.0, 20.0, 20.0, 25.0, 25.0, 25.0}});

    double time = 0.0;
    double inst = 0.0; 
    int incr = 0; 
    std::array<double, 7> current_position;
    std::array<double, 7> current_velocity;
    std::array<double, 7> current_torque;
    
    std::function<franka::Torques(const franka::RobotState&, franka::Duration)> 
    torquescallback = [&](const franka::RobotState& robot_state,
                                                                franka::Duration period) -> franka::Torques {
      time += period.toSec();

      filet >> t;
      
      if (inst/1000 < time) {
        incr = time*1000 - inst;
        inst += incr;
        if (incr >= 1) {
          std::cout << incr << std::endl;
        }
      } else {
        incr = 0;
      }
      
      for (int j=0; j<(1+incr); j++) {
        for (int i=0; i<columns; i++) {
          filetau >> tau[i];
        }
      }
      std::array<double, 7> torque;
      torque = {0.000000, 	 -16.471275, 	 1.455587, 	 16.925426, 	 0.454217, 	 1.884297, 	 -0.000004}; 
      
      franka::Torques waypoint = {{tau[0]-torque[0], tau[1]-torque[1], tau[2]-torque[2], tau[3]-torque[3], tau[4]-torque[4], tau[5]-torque[5], tau[6]-torque[6]}};
      //franka::Torques waypoint = {{tau[0], tau[1], tau[2], tau[3], tau[4], tau[5], tau[6]}};

      inst++;

      if (time >= 5.0) {
        std::cout << std::endl << "Finished motion, shutting down program" << std::endl;
        return franka::MotionFinished(waypoint);
      }
      return waypoint;
    };

    std::function<franka::JointPositions(const franka::RobotState&, franka::Duration)> 
    jointpositionscallback = [&](const franka::RobotState& robot_state,
                                                                franka::Duration period) -> franka::JointPositions {
      time += period.toSec();

      filet >> t;
      
      if (inst/1000 < time) {
        incr = time*1000 - inst;
        inst += incr;
        if (incr >= 1) {
          std::cout << incr << std::endl;
        }
      } else {
        incr = 0;
      }
      
      for (int j=0; j<(1+incr); j++) {
        for (int i=0; i<columns; i++) {
          fileq >> q[i];
        }
      }

      franka::JointPositions waypoint = {{q[0], q[1], q[2], q[3], q[4], q[5], q[6]}};

      inst++;

      current_position = robot_state.q;
      current_velocity = robot_state.dq;
      current_torque = robot_state.tau_J;

      if (stamp_results) {
        if (fileqexp.is_open()){
          for (int i=0; i<columns; i++){
            fileqexp << current_position[i] << '\t';
          }
          fileqexp << std::endl;
        }

        if (fileqpexp.is_open()){
          for (int i=0; i<columns; i++){
            fileqpexp << current_velocity[i] << '\t';
          }
          fileqpexp << std::endl;
        }

        if (filetauexp.is_open()){
          for (int i=0; i<columns; i++){
            filetauexp << current_torque[i] << '\t';
          }
          filetauexp << std::endl;
        }

        if (filetexp.is_open()){
          filetexp << time << std::endl;
        }
      }

      if (time >= 5.0) {
        std::cout << std::endl << "Finished motion, shutting down program" << std::endl;
        return franka::MotionFinished(waypoint);
      }
      return waypoint;
    };

    std::function<franka::JointVelocities(const franka::RobotState&, franka::Duration)> 
    jointvelocitiescallback = [&](const franka::RobotState& robot_state,
                                                                franka::Duration period) -> franka::JointVelocities {
      time += period.toSec();

      filet >> t;
      
      if (inst/1000 < time) {
        incr = time*1000 - inst;
        inst += incr;
        if (incr >= 1) {
          std::cout << incr << std::endl;
        }
      } else {
        incr = 0;
      }
      
      for (int j=0; j<(1+incr); j++) {
        for (int i=0; i<columns; i++) {
          fileqp >> q_p[i];
        }
      }

      franka::JointVelocities waypoint = {{q_p[0], q_p[1], q_p[2], q_p[3], q_p[4], q_p[5], q_p[6]}};

      inst++;

      current_position = robot_state.q;
      current_velocity = robot_state.dq;
      current_torque = robot_state.tau_J;

      if (stamp_results) {
        if (fileqexp.is_open()){
          for (int i=0; i<columns; i++){
            fileqexp << current_position[i] << '\t';
          }
          fileqexp << std::endl;
        }

        if (fileqpexp.is_open()){
          for (int i=0; i<columns; i++){
            fileqpexp << current_velocity[i] << '\t';
          }
          fileqpexp << std::endl;
        }

        if (filetauexp.is_open()){
          for (int i=0; i<columns; i++){
            filetauexp << current_torque[i] << '\t';
          }
          filetauexp << std::endl;
        }

        if (filetexp.is_open()){
          filetexp << time << std::endl;
        }
      }

      if (time >= 5.0) {
        std::cout << std::endl << "Finished motion, shutting down program" << std::endl;
        return franka::MotionFinished(waypoint);
      }
      return waypoint;
    };


    //robot.control(jointpositionscallback);
    robot.control(jointvelocitiescallback);

    fileq.close();
    filetau.close();
    filet.close();
    fileqexp.close();
    filetexp.close();

  } catch (const franka::Exception& e) {
    std::cout << e.what() << std::endl;
    return -1;
  }

  return 0;
}
