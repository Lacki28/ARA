import time
import requests
import datetime
from config import webhooks, central_db_url , logs_url, Microservice1_url
import json
from locust import HttpUser, task, between, events,constant
import csv

# Open CSV file for logging
log_file = open("../response_times.csv", "w", newline="")
csv_writer = csv.writer(log_file)

# Write header
csv_writer.writerow(["timestamp", "request_type", "endpoint", "response_time_ms", "response_length", "status"])

@events.request.add_listener
def log_latency(request_type, name, response_time, response_length, exception, **kwargs):
    status = "OK" if exception is None else "FAIL"
    timestamp = time.time()

    # Print to console
    print(f"{request_type} {name} took {response_time:.2f} ms")

    # Write to CSV
    csv_writer.writerow([
        timestamp,
        request_type,
        name,
        round(response_time, 2),
        response_length,
        status
    ])
    log_file.flush()
class MyUser(HttpUser):
    wait_time = constant(1)#between(1, 3)  # Adjust as needed
    host = Microservice1_url # needs to be added accordingly
    image_data = None  # To store the cached image data

    def save_to_json(self, data):
        with open('timing1.json', 'a') as json_file:
            json.dump(data, json_file)
            json_file.write('\n')

    def on_start(self):
        # Cache image once to avoid concurrent file access issues
        with open('[image_name].jpg', 'rb') as f:
            self.image_data = f.read()


    @task
    def simulate_microservices_flow(self):
        try:
            request_id = str(time.time())
            start_time = datetime.datetime.now()
            locust_start_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{locust_start_time}] Sending request ID: {request_id}")

            headers = {
                'X-Request-ID': request_id,
                'X-Locust-Start-Time': locust_start_time,
                'X-Webhooks': webhooks,
                'X-Central-DB-URL': central_db_url,
                'X-Logs-URL': logs_url,
                'X-Special-Object': 'stop sign'
            }

            response1 = self.client.post(
                "/resize",
                files={'image': ('kermit.jpg', self.image_data, 'image/jpeg')},
                headers=headers,
                timeout=10
            )
            end_time = datetime.datetime.now()
            total_time_ms = (end_time - start_time).total_seconds() * 1000

            request_data = {
                'request_id': request_id,
                'locust_start_time': locust_start_time,
                'response_status_code': response1.status_code,
                'total_time_ms': round(total_time_ms, 2)
            }

            self.save_to_json(request_data)

            if response1.status_code == 200:
                print(f"[{request_id}] Success.")
            else:
                print(f"[{request_id}] Failed with status {response1.status_code}")

        except Exception as e:
            print(f"Exception in task: {e}")