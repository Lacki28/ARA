# Deployment Instructions

To deploy the system, follow the steps from **iContinuum Example 3**:  
https://github.com/disnetlab/iContinuum/tree/master/Example3

## Adjustments Required

### 1. Deployment Images
The updated container images referenced in `deployment.yml.j2` should be adjusted to the following:  

DB: lacki/db:v1

Microservice with resize operation: lacki/microservice1:v1

#### After running the first tests, you can also test it with a different microservice
Microservice with filter operation: lacki/microservice1:v3

### 2. Inventory Configuration
Edit `inventory.invi` and update the placeholder IP addresses with your own IPs.

### 3. Add the adapt agents
Run the compute adapt agent with ```bash python adapt_compute_resources.py``` on the same VM as your Kubernetes Master
Run the network adapt agent with ```bash python adapt_mininet_delay.py``` on the same VM as your Mininet
Add the locust file `locust.py` to where you put the run scripts