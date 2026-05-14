# Track A: Telco Troubleshooting and Optimization Agentic Challenge

## 1. Participant Guide

Participants may deploy the Agent Tool Server locally:

```
# Deploy the server
python server.py

# Run the Agent (example)
python main.py
```

The main.py will output the result.csv in this format:

| ID                                   | Answer |
| ------------------------------------ | ------ |
| 102fb27a-f7a4-480b-ac9b-ef51d7feb912 | C3     |
| 1cb15db5-674a-4586-9ff1-a739e00c98e8 | C5\|C7 |

The questions could be single-answer questions or multiple-answer questions. The answers of multiple-answer questions are divided by "|", e.g., 'C3|C7' or 'C5|C9|C11|C20', in ascending order.

---

## 2. Deployment File Inventory

| File                     | Purpose                            |
|--------------------------| ---------------------------------- |
| `server.py`              | Definition of tools and simulators |
| `utils.py`               | Related functions                  |
| `requirements.txt`       | Python dependencies                |
| `data/Phase_1/test.json` | Question description and choices   |
| `main.py`                | Interface and Agent runner         |
| `results/result.csv`     | output files                       |
