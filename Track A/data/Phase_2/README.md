# Track A: Telco Troubleshooting and Optimization Agentic Challenge

During this phase, participants must use the cloud service provided by the organizers to interact with the simulation sandbox. 
This cloud service deploys the exact same endpoints as in server.py (provided in phase 1).
Two server addresses depending on your location.

## **API Integration & Execution Guide**

### **API Endpoints**

* **Chinese Region (ELB):** https://120.46.145.77/no
* **Overseas Region (Hong Kong ECS):** https://124.71.227.61/no

### **Request Headers**
To access the server, you must use the token bearer provided by Zindi, which must be set in the headers of your request, as below (see example in `main.py`).

```
Content-Type: application/json  
Authorization: Bearer \<Your-Token\>
```