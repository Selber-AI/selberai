#!/bin/bash

while getopts "br" flag; do
  case "$flag" in
    b) BUILD=1;;
    r) RUN=1;;
  esac
done

APP=${@:$OPTIND:1}


if [ $APP = "notebook" ]; then
  
  if [ $BUILD -eq 1 ]; then
    docker build -t app_notebook -f Docker_notebook/Dockerfile .
  fi
  
  if [ $RUN -eq 1 ]; then
    docker run -it --gpus all -p 3333:1111 -v ~/Documents/projects/SelberAI/selberai:/selberai app_notebook
  fi

elif [ $APP = "all_tests" ]; then
  
  if [ $BUILD -eq 1 ]; then
    docker-compose build
  fi

  if [ $RUN -eq 1 ]; then
    docker-compose up
  fi

elif [ $APP = "integration_test" ]; then
  
  if [ $BUILD -eq 1 ]; then
    docker build -t app_integrationtest -f Docker_testintegration/Dockerfile .
  fi

  if [ $RUN -eq 1 ]; then
    docker run -v ~/Documents/projects/SelberAI/selberai:/selberai app_integrationtest
  fi

elif [ $APP = "unit_test" ]; then

  if [ $BUILD -eq 1 ]; then
    docker build -t app_unittest -f Docker_testunit/Dockerfile .
  fi

  if [ $RUN -eq 1 ]; then
    docker run -v ~/Documents/projects/SelberAI/selberai:/selberai app_unittest
  fi

elif [ $APP = "pypi_test" ]; then

  if [ $BUILD -eq 1 ]; then
    docker build -t app_pypitest -f Docker_pypitest/Dockerfile .
  fi

  if [ $RUN -eq 1 ]; then
    docker run app_pypitest
  fi
  
elif [ $APP = "pypi_real" ]; then


  if [ $BUILD -eq 1 ]; then
    docker build -t app_pypi -f Docker_pypi/Dockerfile .
  fi

  if [ $RUN -eq 1 ]; then
    docker run app_pypi
  fi


fi




  
