cmake_minimum_required(VERSION 2.8.6)
project(myplugin C CXX)

# --------------------------------------------
# !  You must set CASADIPATH env var
# !  to the place where you installed casadi.
# --------------------------------------------

# Find the casadi library
find_library(CASADI_LIB casadi 
    $ENV{CASADIPATH}
    $ENV{CASADIPATH}/casadi
REQUIRED)

# Find the casadi headers
find_path(CASADI_INCLUDE casadi.hpp
    $ENV{CASADIPATH}/include/casadi
REQUIRED)

# Set the include directory
include_directories(${CASADI_INCLUDE}/..)

# Build the library
add_library(casadi_nlpsolver_mysolver SHARED
  mysolver_interface.cpp mysolver_interface.hpp)

# Make sure our new library links to the casadi library
target_link_libraries(casadi_nlpsolver_mysolver ${CASADI_LIB})
