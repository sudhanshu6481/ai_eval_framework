# GFiber ADK Agent Evaluation Framework

An automated, highly scalable routing intent validation and metrics extraction test suite built on top of the **pytest** framework. This project evaluates end-to-end intent matching performance against the Google Vertex AI Platform Reasoning Engine.

---

## 📋 Table of Contents
1. [Project Structure](#project-structure)
2. [Prerequisites & Environment Variables](#prerequisites--environment-variables)
3. [Authentication & Setup](#authentication--setup)
4. [Installation](#installation)
5. [Running the Project](#running-the-project)
6. [Understanding the Output](#understanding-the-output)

---

## 📁 Project Structure

```text
gFiber_eval_framework/
├── .env                     # Local environment configurations (Excluded from VCS)
├── business_baseline.json   # Comprehensive test cases matrix source
├── test_orchestrator.py     # Core pytest automation test suite
└── test_results.csv         # Automatically generated evaluation reports matrix

Prerequisites & Environment Variables
Create a .env file in the root directory (gFiber_eval_framework/) to store your Google Cloud Platform and test file configurations:

# GCP Project Details
GCP_PROJECT_ID=gfp-d-ai-agents
GCP_LOCATION=us-central1

# Master Orchestrator Application Entrypoint
GCP_ORCHESTRATOR_ID=7827672317521035264

# Test File Configurations
INPUT_JSON=business_baseline.json
OUTPUT_CSV=test_results.csv

Authentication & Setup
If you are operating within a corporate workspace or on an internal managed machine, your credentials and tokens must be initialized to safely pull packages and hit Google Cloud APIs.

1. Refresh Security SSO Tickets
Before installing dependencies or calling cloud utilities, fetch a fresh Single Sign-On (SSO) credential ticket to unblock local package mirrors:

gcert

Note: This command may prompt an authentication flow within your web browser or require your security key.

2. Authenticate Google Cloud SDK (gcloud)
Authorize your active shell account pool to access the targeted Google Cloud infrastructure project:

gcloud auth login

Verify your active working workspace configuration profile:

gcloud config set project gfp-d-ai-agents

3. Installation
Once your security tickets are valid, install the required packages using the explicit Python 3 pip client manager module:

pip3 install pytest pandas python-dotenv google-cloud-aiplatform

Note on pip executable errors: If your terminal returns a command not found exception for pip, route the call explicitly through Python's internal executable module layer:

Bash
python3 -m pip install pytest pandas python-dotenv google-cloud-aiplatform

4. Running the Project
Ensure that your business_baseline.json file is correctly placed inside your local directory alongside the scripts. Run the execution engine runner test runner suite using the following optimized execution flags:

Bash
python3 -m pytest test_orchestrator.py -v -s --cache-clear