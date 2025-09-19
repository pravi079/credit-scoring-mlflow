import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = Path("data/raw/bank.csv")

def load_data():
    """Load training and test datasets."""
    df = pd.read_csv(RAW_DATA_PATH,sep=";")

   # Features vs target
    X = df.drop(columns=["y"])
    y = df["y"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    x_train, x_test,y_train, y_test = load_data()
    print(f"here is the x_train - ,{x_train}")
    print(f"here is the x_train - ,{x_test}")
    print(f"here is the x_train - ,{y_train}")
    print(f"here is the x_train - ,{y_test}")
    