#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
compose_file="$SCRIPT_DIR/../train-compose.yml"
export RUN_ID=${1:-""}
docker compose -f $compose_file up carla-server -d
sleep 5
docker compose -f $compose_file up training -d