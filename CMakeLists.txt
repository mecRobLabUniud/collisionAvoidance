cmake_minimum_required(VERSION 3.5)
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
add_executable(robot_control
    src/robot_control.cpp
    src/examples_common.cpp
)

target_link_libraries(robot_control PRIVATE
    Franka::Franka
    Eigen3::Eigen
    ${ZMQ_LIBRARIES}
    Threads::Threads
    # casadi
    # pinocchio::pinocchio
)