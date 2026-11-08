from pathlib import Path
from urllib.request import urlretrieve

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Public mirror of the commonly used Pima Indians Diabetes dataset.
URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
DESTINATION = DATA_DIR / "diabetes.csv"

if DESTINATION.exists():
    print(f"Dataset already exists: {DESTINATION}")
else:
    print("Downloading dataset...")
    urlretrieve(URL, DESTINATION)
    print(f"Saved dataset to: {DESTINATION}")
