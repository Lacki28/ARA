from flask import Flask, request, jsonify
from kubernetes import client, config
import json
import os
import logging
from config import k3s_config_file

app = Flask(__name__)
logging.basicConfig(filename='k8s_management.log', level=logging.INFO)

# === Shared Initialization ===
def load_k8s_config():
    if os.path.exists(k3s_config_file) and os.access(k3s_config_file, os.R_OK):
        config.load_kube_config(config_file=k3s_config_file)
        return True
    else:
        logging.error("K3s config missing or not readable")
        return False

# === Route: Update Replicas ===
@app.route("/k8s/scale", methods=["POST"])
def scale_deployment():
    data = request.get_json()
    try:
        deployment_name = data["deployment_name"]
        namespace = data.get("namespace", "default")
        new_replicas = int(data["new_replicas"])

        if not load_k8s_config():
            return jsonify({"error": "K3s config not loaded"}), 500

        apps_api = client.AppsV1Api()
        body = {"spec": {"replicas": new_replicas}}
        apps_api.patch_namespaced_deployment(name=deployment_name, namespace=namespace, body=body)

        msg = f"Updated {deployment_name} to {new_replicas} replicas."
        logging.info(msg)
        return jsonify({"status": "success", "message": msg})
    except Exception as e:
        logging.error(str(e))
        return jsonify({"status": "error", "message": str(e)}), 500

# === Route: Update CPU/Memory Limits ===
@app.route("/k8s/resources", methods=["POST"])
def update_resources():
    data = request.get_json()
    try:
        deployment_name = data["deployment_name"]
        namespace = data.get("namespace", "default")
        cpu_limit = data["new_cpu_limit"]
        memory_limit = data["new_memory_limit"]

        if not load_k8s_config():
            return jsonify({"error": "K3s config not loaded"}), 500

        apps_api = client.AppsV1Api()
        deployment = apps_api.read_namespaced_deployment(deployment_name, namespace)

        for container in deployment.spec.template.spec.containers:
            if not container.resources.limits:
                container.resources.limits = {}
            if not container.resources.requests:
                container.resources.requests = {}
            container.resources.limits["cpu"] = cpu_limit
            container.resources.limits["memory"] = memory_limit
            container.resources.requests["cpu"] = "100m"
            container.resources.requests["memory"] = memory_limit

        apps_api.patch_namespaced_deployment(deployment_name, namespace, deployment)

        msg = f"Updated resources for {deployment_name} to CPU: {cpu_limit}, MEM: {memory_limit}"
        logging.info(msg)
        print(msg)
        return jsonify({"status": "success", "message": msg})
    except Exception as e:
        logging.error(str(e))
        print(str(e))
        return jsonify({"status": "error", "message": str(e)}), 500

# === Route: Get Cluster Info ===
@app.route("/k8s/cluster-info", methods=["GET"])
def get_cluster_info():
    try:
        if not load_k8s_config():
            return jsonify({"error": "K3s config not loaded"}), 500

        core_api = client.CoreV1Api()
        apps_api = client.AppsV1Api()
        node_list = core_api.list_node()
        pod_list = core_api.list_pod_for_all_namespaces()
        deployment_list = apps_api.list_deployment_for_all_namespaces()

        result = {
            "nodes": [node.metadata.name for node in node_list.items],
            "deployments": [
                {"name": d.metadata.name, "replicas": d.spec.replicas}
                for d in deployment_list.items
            ],
            "pods": [
                {"name": p.metadata.name, "node": p.spec.node_name}
                for p in pod_list.items if p.metadata.name.startswith("microservice")
            ]
        }

        return jsonify(result)
    except Exception as e:
        logging.error(str(e))
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050)
