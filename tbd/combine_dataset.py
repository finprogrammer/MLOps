import pandas as pd
import os

class DatasetCombiner:
    def __init__(self, train_path: str, test_path: str, output_path: str, mark_source: bool = True):
        self.train_path = train_path
        self.test_path = test_path
        self.output_path = output_path
        self.mark_source = mark_source

    def combine(self):
        print(f"Reading training data from: {self.train_path}")
        train_df = pd.read_csv(self.train_path)

        print(f"Reading testing data from: {self.test_path}")
        test_df = pd.read_csv(self.test_path)

        if self.mark_source:
            train_df["source"] = "train"
            test_df["source"] = "test"

        print("Combining datasets...")
        combined_df = pd.concat([train_df, test_df], ignore_index=True)

        print(f"Saving combined dataset to: {self.output_path}")
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        combined_df.to_csv(self.output_path, index=False)

        print(f"Done. Combined dataset shape: {combined_df.shape}")
        return combined_df

def main():
    # Modify these paths as needed
    train_csv = "Network_Data/UNSW_NB15_training-set.csv"
    test_csv = "Network_Data/UNSW_NB15_testing-set.csv"
    output_csv = "Network_Data/UNSW_NB15.csv"

    combiner = DatasetCombiner(train_csv, test_csv, output_csv, mark_source=False)
    combiner.combine()

if __name__ == "__main__":
    main()
