# Deployment Instructions

To deploy the system, follow the steps from **iContinuum Example 3**:  
https://github.com/disnetlab/iContinuum/tree/master/Example3

## Adjustments Required

### 1. Deployment Images
The updated container images referenced in `deployment.yml.j2` should be adjusted to the following:  
DB: cvetac/latest2
Microservice with resize operation: lacki/microservice1:v1

#### After running the first tests, you can also test it with a different microservice
Microservice with filter operation: lacki/microservice1:v3

### 2. Inventory Configuration
Edit `inventory.invi` and update the placeholder IP addresses with your own IPs.