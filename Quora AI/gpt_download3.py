import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'   # 👈 add this line
import os
import requests
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# 🔇 Suppress TensorFlow backend logs (the oneDNN messages)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def download_and_load_gpt2(model_size, models_dir):
    """
    Download and load GPT-2 model weights from OpenAI’s public storage.

    Args:
        model_size (str): One of ("124M", "355M", "774M", "1558M").
        models_dir (str): Directory to save model files.

    Returns:
        settings (dict): GPT-2 hyperparameters.
        params (dict): GPT-2 model weights converted from TensorFlow.
    """
    # Validate model size
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size must be one of {allowed_sizes}")

    # Directory for this model
    model_dir = os.path.join(models_dir, model_size)
    os.makedirs(model_dir, exist_ok=True)

    # Base URL for GPT-2 weights
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint",
        "encoder.json",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "vocab.bpe"
    ]

    # ==========================================================
    # Download files if missing
    # ==========================================================
    print(f"🔍 Checking GPT-2 ({model_size}) files in: {model_dir}")

    for filename in filenames:
        file_url = f"{base_url}/{model_size}/{filename}"
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path)

    print("✅ All model files are present.")

    # ==========================================================
    # Load model settings and weights
    # ==========================================================
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    if not tf_ckpt_path:
        raise FileNotFoundError("No TensorFlow checkpoint found in model directory.")

    settings = json.load(open(os.path.join(model_dir, "hparams.json"), encoding="utf-8"))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    print("✅ GPT-2 weights loaded successfully.")
    return settings, params


def download_file(url, destination):
    """
    Download a file from a URL if it doesn't exist locally.
    """
    try:
        # Check if file already exists
        if os.path.exists(destination):
            print(f"✔️  File exists: {os.path.basename(destination)}")
            return

        # Request file
        response = requests.get(url, stream=True, verify=False)
        response.raise_for_status()

        # File size for progress bar
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024

        # Progress bar
        desc = os.path.basename(destination)
        with tqdm(total=total_size, unit="iB", unit_scale=True, desc=desc) as bar:
            with open(destination, "wb") as f:
                for chunk in response.iter_content(block_size):
                    if chunk:
                        bar.update(len(chunk))
                        f.write(chunk)

        print(f"✅ Downloaded: {os.path.basename(destination)}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading {url}: {e}")


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    """
    Convert TensorFlow GPT-2 checkpoint weights into a PyTorch-compatible dictionary.
    """
    print("🔄 Converting TensorFlow checkpoint to PyTorch format...")

    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    for name, _ in tf.train.list_variables(ckpt_path):
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))
        parts = name.split("/")[1:]  # Skip 'model/'

        target_dict = params
        if parts[0].startswith("h"):  # Transformer block
            layer_number = int(parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access nested dicts
        for key in parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        last_key = parts[-1]
        target_dict[last_key] = variable_array

    print("✅ Conversion complete.")
    return params


# ==========================================================
# Run directly (manual test)
# ==========================================================
if __name__ == "__main__":
    model_size = "355M"
    models_dir = "./gpt2"
    settings, params = download_and_load_gpt2(model_size, models_dir)
    print("\n🎉 GPT-2 model successfully downloaded and loaded.")





















'''import os
import requests  # Make sure requests is installed
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def download_and_load_gpt2(model_size, models_dir):
    # Validate model size
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Define paths
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # Download files
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path)

    ## We have reached here until now ---> we have downloaded the files on our local machine.

    # Load settings and params
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params

def download_file(url, destination):
    try:
        # Send a GET request to download the file, disabling SSL verification
        response = requests.get(url, stream=True, verify=False)

        # Get the total file size from headers, defaulting to 0 if not present
        file_size = int(response.headers.get("content-length", 0))

        # Check if file exists and has the same size
        if os.path.exists(destination):
            file_size_local = os.path.getsize(destination)
            if file_size == file_size_local:
                print(f"File already exists and is up-to-date: {destination}")
                return

        # Define the block size for reading the file
        block_size = 1024  # 1 Kilobyte

        # Initialize the progress bar with total file size
        progress_bar_description = url.split("/")[-1]  # Extract filename from URL
        with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
            # Open the destination file in binary write mode
            with open(destination, "wb") as file:
                # Iterate over the file data in chunks
                for chunk in response.iter_content(block_size):
                    progress_bar.update(len(chunk))  # Update progress bar
                    file.write(chunk)  # Write the chunk to the file

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
        print(f"Please check the URL: {url}")

def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # Initialize parameters dictionary with empty blocks for each layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params

if __name__ == "__main__":
    model_size = "355M"  # Can be "355M", "774M", "1558M"
    models_dir = "./models"  # The folder where files should be downloaded
    settings, params = download_and_load_gpt2(model_size, models_dir)
    print("✅ Download and loading complete.")'''
