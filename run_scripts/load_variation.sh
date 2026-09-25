#!/bin/bash
bash kill.sh

ROUNDS=2

for ((i=1; i<=ROUNDS; i++)); do
    python ../reset.py
    rm timing1.json
    curl -X POST http://172.16.0.101:32702/clear_data
    locust -f ../locust/locustfile.py --headless -u 10 -r 1 --run-time "4340"&
    PID=$!
    echo "Locust started with PID $PID"10
    python dict.py --network-lower 1000 --network-upper 1500 --compute-lower 1000 --compute-upper 1500 --wandb_name "MVP change users 10 to 5" --strategy "proactive" &
    PID2=$!
    echo "Coordination started with PID $PID2"
    sleep 4340
    locust -f ../locust/locustfile.py --headless -u 5 -r 1 --run-time "4340"&
    sleep 4340
    bash kill.sh
    sleep 30
    bash kill.sh
done


for ((i=1; i<=ROUNDS; i++)); do
    python ../reset.py
    rm timing1.json
    curl -X POST http://172.16.0.101:32702/clear_data
    locust -f ../locust/locustfile.py --headless -u 10 -r 1 --run-time "4340"&
    PID=$!
    echo "Locust started with PID $PID"
    python dict.py --network-lower 1000 --network-upper 1500 --compute-lower 1000 --compute-upper 1500 --wandb_name "MVP change users 10 to 5" --strategy "SA_PPO" &
    PID2=$!
    echo "Coordination started with PID $PID2"
    sleep 4340
    locust -f ../locust/locustfile.py --headless -u 5 -r 1 --run-time "4340"&
    sleep 4340
    bash kill.sh
    sleep 30
    bash kill.sh
done

for ((i=1; i<=ROUNDS; i++)); do
    python ../reset.py
    rm timing1.json
    curl -X POST http://172.16.0.101:32702/clear_data
    locust -f ../locust/locustfile.py --headless -u 10 -r 1 --run-time "4340"&
    PID=$!
    echo "Locust started with PID $PID"
    python dict.py --network-lower 1000 --network-upper 1500 --compute-lower 1000 --compute-upper 1500 --wandb_name "MVP change users 10 to 5" --strategy "reactive" &
    PID2=$!
    echo "Coordination started with PID $PID2"
    sleep 4340
    locust -f ../locust/locustfile.py --headless -u 5 -r 1 --run-time "4340"&
    sleep 4340
    bash kill.sh
    sleep 30
    bash kill.sh
done

for ((i=1; i<=ROUNDS; i++)); do
    python ../reset.py
    rm timing1.json
    curl -X POST http://172.16.0.101:32702/clear_data
    locust -f ../locust/locustfile.py --headless -u 10 -r 1 --run-time "4340"&
    PID=$!
    echo "Locust started with PID $PID"
    python dict.py --network-lower 1000 --network-upper 1500 --compute-lower 1000 --compute-upper 1500 --wandb_name "MVP change users 10 to 5" --strategy "PPO" &
    PID2=$!
    echo "Coordination started with PID $PID2"
    sleep 4340
    locust -f ../locust/locustfile.py --headless -u 5 -r 1 --run-time "4340"&
    sleep 4340
    bash kill.sh
    sleep 30
    bash kill.sh
done