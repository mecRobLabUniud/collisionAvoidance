cmake_minimum_required(VERSION 3.10)
project(CollisionAvoidance CXX)

# Richiesto C++14 per libfranka e Pinocchio
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- DIPENDENZE ---
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Franka 0.8.0 REQUIRED)
find_package(Threads REQUIRED)

# ZeroMQ
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED libzmq)

# CasADi e Pinocchio
# find_package(casadi REQUIRED)
# find_package(pinocchio REQUIRED)

# --- INCLUDE ---
include_directories(
    ${ZMQ_INCLUDE_DIRS}
    include
)

# --- ESEGUIBILE ---
add_executable(robot_control src/robot_control.cpp src/examples_common.cpp)
target_link_libraries(robot_control PRIVATE Franka::Franka Eigen3::Eigen ${ZMQ_LIBRARIES} Threads::Threads)

add_executable(trajectory_interpolation src/trajectory_interpolation.cpp)
target_link_libraries(trajectory_interpolation PRIVATE Eigen3::Eigen)
# target_compile_options(trajectory_interpolation PRIVATE -O2)

add_executable(joint_point_to_point_motion src/joint_point_to_point_motion.cpp src/examples_common.cpp)
target_link_libraries(joint_point_to_point_motion PRIVATE Franka::Franka Eigen3::Eigen)

add_executable(record_trajectory src/record_trajectory.cpp)
target_link_libraries(record_trajectory PRIVATE Franka::Franka Eigen3::Eigen Threads::Threads)

add_executable(replay_trajectory src/replay_trajectory.cpp src/examples_common.cpp)
target_link_libraries(replay_trajectory PRIVATE Franka::Franka Eigen3::Eigen)

