#!/usr/bin/env bash

## Complete the following steps to get Docker running locally

# Step 1:
docker build . --tag "project-ml-microservice:latest"

# Step 2: 
docker image list

# Step 3: 
docker run -p 8000:80 project-ml-microservice:latest
