"""
This script downloads the BankSim dataset from datahub.io.

The dataset is used for simulating bank transactions and is often employed
in fraud detection and other machine learning tasks related to financial data.
"""

import requests


def download_banksim():
    """
    Downloads the BankSim dataset from a specified URL and saves it locally.

    The function performs the following steps:
    1. Defines the URL for the BankSim dataset and the local path for saving.
    2. Sends an HTTP GET request to the URL.
    3. Checks if the request was successful (HTTP status 200).
    4. Writes the downloaded content to a local file in chunks.
    5. Prints status messages during the download and upon completion.
    """
    # URL from which to download the BankSim dataset
    url = "https://datahub.io/machine-learning/banksim/r/banksim.csv"
    # Local file name to save the downloaded dataset
    local_path = "banksim.csv"
    print(f"Downloading BankSim dataset from {url} ...")
    # Send a GET request to the URL. stream=True allows downloading large files efficiently.
    r = requests.get(url, stream=True)
    # Raise an HTTPError if the HTTP request returned an unsuccessful status code.
    r.raise_for_status()
    # Open the local file in write-binary ('wb') mode to save the content.
    with open(local_path, "wb") as f:
        # Iterate over the response data in chunks of 8KB.
        # This is memory-efficient for large files.
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Saved to {local_path}")


# This block ensures that download_banksim() is called only when the script is executed directly,
# not when it's imported as a module into another script.
if __name__ == "__main__":
    download_banksim()
