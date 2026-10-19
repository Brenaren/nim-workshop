#!/bin/bash
# Stop all docker containers (except nginx server)
docker ps --format '{{.Names}} {{.ID}}' | grep -v "nginx" | grep -v "docker_nemo"  | awk '{print $2}' | xargs -r docker kill
docker ps -a --format '{{.Names}} {{.ID}}' | grep -v "nginx" | grep -v "docker_nemo" |  awk '{print $2}' | xargs -r docker rm


