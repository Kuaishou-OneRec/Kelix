RANK=$OMPI_COMM_WORLD_RANK
head_node_addr=$1
port=$2

if [[ $RANK -eq 0 ]]; then
    echo 'Start ray head'
    ray start --head --port=$port
else
    sleep 5
    echo 'Add ray node'
    ray start --address=$head_node_addr:$port
fi

sleep 5

ray status

