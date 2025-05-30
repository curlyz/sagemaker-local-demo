import requests


def download_banksim():
    url = "https://datahub.io/machine-learning/banksim/r/banksim.csv"
    local_path = "banksim.csv"
    print(f"Downloading BankSim dataset from {url} ...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(local_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Saved to {local_path}")


if __name__ == "__main__":
    download_banksim()
