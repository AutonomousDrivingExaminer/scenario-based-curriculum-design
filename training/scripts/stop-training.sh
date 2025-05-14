#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
compose_file="$SCRIPT_DIR/../train-compose.yml"
docker compose -f $compose_file down