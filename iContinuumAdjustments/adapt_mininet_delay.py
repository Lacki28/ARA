from flask import Flask, request, jsonify
import subprocess

app = Flask(__name__)

# List of Mininet interfaces to apply delay on
interfaces = [
    's1-eth1', 's1-eth2', 's1-eth3', 's1-eth4', 's1-eth5',
    's2-eth1', 's2-eth2', 's2-eth3', 's2-eth4','s2-eth5',
    's3-eth1', 's3-eth2', 's3-eth3', 's3-eth4',
    's4-eth1', 's4-eth2', 's4-eth3', 's4-eth4',
    's5-eth1', 's5-eth2',
    's6-eth1', 's6-eth2'
]

@app.route('/network/delay', methods=['POST'])
def update_delay():
    try:
        data = request.get_json()

        # If no JSON or no delay value → do nothing
        if not data or "delay_ms" not in data:
            print("→ No delay value provided. Ignoring request.")
            return jsonify({"status": "ignored", "reason": "no delay_ms provided"}), 200

        delay_ms = int(data["delay_ms"])

        for intf in interfaces:
            print(f"→ Updating {intf} to {delay_ms}ms")
            subprocess.run(f"tc qdisc del dev {intf} root", shell=True)
            subprocess.run(f"tc qdisc add dev {intf} root netem delay {delay_ms}ms", shell=True)

        return jsonify({"status": "success", "delay_ms": delay_ms})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500



@app.route('/network/bandwidth', methods=['POST'])
def update_bandwidth():
    try:
        data = request.get_json()
        rate = data.get("rate", "10mbit")   # default 10mbit
        burst = data.get("burst", "64kbit")
        latency = data.get("latency", "400ms")

        for intf in interfaces:
            print(f"→ Limiting {intf} to {rate}")
            subprocess.run(f"tc qdisc del dev {intf} root", shell=True)
            subprocess.run(
                f"tc qdisc add dev {intf} root tbf rate {rate} burst {burst} latency {latency}",
                shell=True
            )

        return jsonify({"status": "success", "rate": rate})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/network/bandwidth', methods=['GET'])
def get_bandwidth():
    results = {}
    try:
        for intf in interfaces:
            output = subprocess.check_output(
                f"tc qdisc show dev {intf}", shell=True, text=True
            )
            results[intf] = output.strip()
        return jsonify({"status": "success", "bandwidth": results})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5050)
