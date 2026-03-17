cmake_minimum_required(VERSION 3.5)
project(controllo_capsule_dinamiche CXX)

# Richiesto C++14 per libfranka e Pinocchio
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- DIPENDENZE ---
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Franka REQUIRED)
find_package(Threads REQUIRED)

# ZeroMQ
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED libzmq)

# CasADi e Pinocchio
find_package(casadi REQUIRED)
find_package(pinocchio REQUIRED)

# --- INCLUDE ---
include_directories(
    ${ZMQ_INCLUDE_DIRS}
    .
)

# --- ESEGUIBILE ---
add_executable(controllo_pick_and_place_capsule_dinamiche_parallelo
    controllo_pick_and_place_capsule_dinamiche_parallelo.cpp
    header_capsuleDinamiche.cpp
    examples_common.cpp
)

target_link_libraries(controllo_pick_and_place_capsule_dinamiche_parallelo PRIVATE
    Franka::Franka
    Eigen3::Eigen
    ${ZMQ_LIBRARIES}
    Threads::Threads
    casadi
    pinocchio::pinocchio
)