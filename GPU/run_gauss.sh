#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
/bin/echo "##" $(whoami) is compiling gauss
dpcpp gauss.cpp -o gauss -O3 -Wall
if [ $? -eq 0 ]; then ./gauss; fi