#!/bin/bash
host=$1
start=$2
end=$3
ssh $host $(for i in `seq ${start} ${end}` ;do echo -L $i:localhost:$i ;done)
