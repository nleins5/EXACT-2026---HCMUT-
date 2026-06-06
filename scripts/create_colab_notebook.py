import json
from pathlib import Path

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# EXACT 2026 Live API - Google Colab Deployment\n",
                "\n",
                "This notebook allows you to host the EXACT-2026 API server on Google Colab using a free T4 GPU. This prevents you from having to keep your local machine running 24/7 during the Phase 2 evaluation window.\n",
                "\n",
                "### ⚠️ Instructions:\n",
                "1. Go to **Runtime** -> **Change runtime type** and select **T4 GPU** (to enable CUDA acceleration for `llama-server`).\n",
                "2. Fill in the configuration form below.\n",
                "3. Click **Runtime** -> **Run all**.\n",
                "4. Keep this browser tab open to prevent Colab from sleeping."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "#@title Configuration Form\n",
                "REPO_URL = \"https://github.com/nleins5/EXACT-2026---HCMUT-.git\" #@param {type:\"string\"}\n",
                "NGROK_AUTHTOKEN = \"REMOVED_NGROK_AUTHTOKEN\" #@param {type:\"string\"}\n",
                "PORT = 8000 #@param {type:\"integer\"}"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Setup Environment and Clone Repository"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Clone the repository\n",
                "import os\n",
                "import shutil\n",
                "\n",
                "PROJECT_NAME = REPO_URL.split(\"/\")[-1].replace(\".git\", \"\")\n",
                "if os.path.exists(PROJECT_NAME):\n",
                "    print(f\"Removing existing {PROJECT_NAME} directory...\")\n",
                "    shutil.rmtree(PROJECT_NAME)\n",
                "\n",
                "!git clone {REPO_URL}\n",
                "%cd {PROJECT_NAME}"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Build llama-server with CUDA"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Clone llama.cpp and compile llama-server with CUDA support\n",
                "%cd /content\n",
                "if not os.path.exists(\"llama.cpp\"):\n",
                "    !git clone --depth 1 https://github.com/ggerganov/llama.cpp.git\n",
                "\n",
                "%cd llama.cpp\n",
                "!cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=OFF\n",
                "!cmake --build build --config Release -t llama-server -j\n",
                "\n",
                "# Link llama-server to PATH\n",
                "!ln -sf /content/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server\n",
                "print(\"\\nllama-server compiled and linked successfully!\")\n",
                "!llama-server --version"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Install Dependencies & Download Models"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%cd /content/{PROJECT_NAME}\n",
                "# Install requirements\n",
                "!pip install -r requirements.txt\n",
                "\n",
                "# Download GGUF models\n",
                "!python3 models/download_models.py"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Install and Configure ngrok"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Install official ngrok package\n",
                "!curl -s https://ngrok-agent.s3.amazonaws.com/files.fee.sh/harbinger/setup.sh | sudo bash\n",
                "!sudo apt-get install ngrok\n",
                "\n",
                "# Add authtoken\n",
                "!ngrok config add-authtoken {NGROK_AUTHTOKEN}"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Launch API Server and Expose Tunnel"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import subprocess\n",
                "import time\n",
                "import json\n",
                "import urllib.request\n",
                "\n",
                "# Start the FastAPI server in the background\n",
                "print(\"Starting FastAPI server...\")\n",
                "api_process = subprocess.Popen(\n",
                "    [\"python3\", \"-m\", \"uvicorn\", \"src.api.app:app\", \"--host\", \"0.0.0.0\", \"--port\", str(PORT)],\n",
                "    stdout=subprocess.PIPE,\n",
                "    stderr=subprocess.STDOUT,\n",
                "    text=True\n",
                ")\n",
                "\n",
                "# Wait for API to initialize\n",
                "time.sleep(5)\n",
                "\n",
                "# Start ngrok tunnel in the background\n",
                "print(\"Starting ngrok tunnel...\")\n",
                "ngrok_process = subprocess.Popen(\n",
                "    [\"ngrok\", \"http\", str(PORT), \"--log\", \"stdout\"],\n",
                "    stdout=subprocess.PIPE,\n",
                "    stderr=subprocess.STDOUT,\n",
                "    text=True\n",
                ")\n",
                "\n",
                "time.sleep(5)\n",
                "\n",
                "# Retrieve ngrok public URL\n",
                "try:\n",
                "    with urllib.request.urlopen(\"http://localhost:4040/api/tunnels\") as response:\n",
                "        data = json.loads(response.read().decode())\n",
                "        public_url = data[\"tunnels\"][0][\"public_url\"]\n",
                "        print(\"\\n\" + \"=\"*50)\n",
                "        print(f\"🚀 EXACT 2026 API IS LIVE!\")\n",
                "        print(f\"Public URL: {public_url}\")\n",
                "        print(f\"Health Check: {public_url}/health\")\n",
                "        print(f\"Models Info: {public_url}/v1/models\")\n",
                "        print(\"=\"*50 + \"\\n\")\n",
                "except Exception as e:\n",
                "    print(f\"Failed to get ngrok public URL: {e}\")\n",
                "    print(\"Please check if ngrok authtoken is correct.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Monitor and Uptime Keep-Alive\n",
                "This cell runs indefinitely to display incoming request logs and keeps the Colab runtime active."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import select\n",
                "import sys\n",
                "\n",
                "print(\"Monitoring FastAPI logs... (Press Stop in Colab to terminate)\")\n",
                "\n",
                "try:\n",
                "    while True:\n",
                "        # Read line-by-line from FastAPI output\n",
                "        line = api_process.stdout.readline()\n",
                "        if line:\n",
                "            sys.stdout.write(line)\n",
                "            sys.stdout.flush()\n",
                "        else:\n",
                "            # If process ended\n",
                "            rc = api_process.poll()\n",
                "            if rc is not None:\n",
                "                print(f\"FastAPI process exited with code {rc}\")\n",
                "                break\n",
                "        time.sleep(0.01)\n",
                "except KeyboardInterrupt:\n",
                "    print(\"Stopping servers...\")\n",
                "    api_process.terminate()\n",
                "    ngrok_process.terminate()"
            ]
        }
    ],
    "metadata": {
        "accelerator": "GPU",
        "gpuClass": "standard",
        "colab": {
            "provenance": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

# Write notebook to disk
with open("EXACT_2026_Colab_Deployment.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated successfully!")
