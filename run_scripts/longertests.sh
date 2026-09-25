#!/bin/bash
bash kill.sh
ROUNDS=1
sleep 30
bash kill.sh

STRATEGIES=("SA_PPO", "reactive", "PPO", "proactive")

for ((i=1; i<=ROUNDS; i++)); do
    for strategy in "${STRATEGIES[@]}"; do
        python ../reset.py
        rm timing1.json
        curl -X POST http://172.16.0.101:32702/clear_data

        # Loop through strategies
        echo "Running strategy: $strategy"

        locust -f ../locust/locustfile.py --headless -u 10 -r 1 --run-time "4340" &
        PID=$!
        echo "Locust started with PID $PID"

        python dict.py \
            --network-lower 1000 \
            --network-upper 1500 \
            --compute-lower 1000 \
            --compute-upper 1500 \
            --wandb_name "ARA long runs" \
            --strategy "$strategy" &

        PID2=$!
        echo "Coordination started with PID $PID2"

        sleep 4340
        locust -f ../locust/locustfile.py --headless -u 5 -r 1 --run-time "4340" &
        python cp_1000_2000.py
        sleep 4340

        # Your remaining fixed sequence
        locust -f ../locust/locustfile.py --headless -u 5 -r 1 --run-time "4340" &
        python nw_250_750.py
        sleep 4340

        locust -f ../locust/locustfile.py --headless -u 5 -r 1 --run-time "4340" &
        python cp_250_750.py
        sleep 4340

        locust -f ../locust/locustfile.py --headless -u 15 -r 1 --run-time "4340" &
        python request_resetter.py
        sleep 4340

        bash kill.sh
        sleep 30
        bash kill.sh
    done
done
