#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
compose_file="$SCRIPT_DIR/../train-compose.yml"
RUN_ID= docker compose -f $compose_file logs -f training