import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder

class DataAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_csv(file_path)
        self.analysis_results = {}

    def analyze(self, target_column=None, output_dir='eda_report'):
        os.makedirs(output_dir, exist_ok=True)

        missing = self.df.isnull().sum()
        missing_percent = 100 * missing / len(self.df)
        missing_df = pd.DataFrame({
            'Missing Values': missing,
            'Percentage': missing_percent
        }).sort_values(by='Missing Values', ascending=False)
        self.analysis_results['missing_values'] = missing_df
        missing_df.to_csv(os.path.join(output_dir, 'missing_values.csv'))


        placeholder_values = ["?", "unknown", "none", "null",]
        placeholder_report = {}
        for col in self.df.columns:
            for val in placeholder_values:
                count = (self.df[col].astype(str).str.lower() == val).sum()
                if count > 0:
                    placeholder_report.setdefault(col, {})[val] = count

        placeholder_df = pd.DataFrame.from_dict(placeholder_report, orient='index').fillna(0).astype(int)
        placeholder_df.to_csv(os.path.join(output_dir, 'placeholder_values.csv'))
        self.analysis_results['placeholder_missing'] = placeholder_df

        numeric_df = self.df.select_dtypes(include=[np.number])
        z_scores = np.abs(zscore(numeric_df.dropna()))
        outliers = (z_scores > 3).sum(axis=0)
        outliers_df = pd.DataFrame({
            'Outlier Count': outliers,
            'Percentage': 100 * outliers / len(numeric_df)
        }, index=numeric_df.columns)  

        self.analysis_results['outliers'] = outliers_df
        outliers_df.to_csv(os.path.join(output_dir, 'outliers.csv'))


        for col in outliers_df[outliers_df['Percentage'] > 1].index:
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=self.df[col])
            plt.title(f'Outliers in {col}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'boxplot_{col}.png'))
            plt.close()


        skewness = numeric_df.skew().sort_values(ascending=False)
        skew_df = pd.DataFrame({'Skewness': skewness})
        self.analysis_results['distribution_skewness'] = skew_df
        skew_df.to_csv(os.path.join(output_dir, 'distribution_skewness.csv'))

        corr_matrix = numeric_df.corr()
        self.analysis_results['correlation_matrix'] = corr_matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
        plt.close()

        cat_df = self.df.select_dtypes(include=['object', 'category'])
        cat_cardinality = cat_df.nunique().sort_values(ascending=False)
        cat_card_df = pd.DataFrame({'Unique Categories': cat_cardinality})
        self.analysis_results['categorical_cardinality'] = cat_card_df
        cat_card_df.to_csv(os.path.join(output_dir, 'categorical_cardinality.csv'))

        if target_column and target_column in self.df.columns:
            target_counts = self.df[target_column].value_counts()
            target_df = pd.DataFrame({'Count': target_counts, 'Percentage': 100 * target_counts / len(self.df)})
            self.analysis_results['target_imbalance'] = target_df
            target_df.to_csv(os.path.join(output_dir, 'target_distribution.csv'))
            plt.figure(figsize=(8, 6))
            sns.countplot(x=self.df[target_column], order=target_counts.index)
            plt.title("Target Variable Distribution")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'target_distribution.png'))
            plt.close()

        duplicate_count = self.df.duplicated().sum()
        self.analysis_results['duplicate_rows'] = duplicate_count
        with open(os.path.join(output_dir, 'duplicate_info.txt'), 'w') as f:
            f.write(f'Total Duplicate Rows: {duplicate_count}')

        for col in numeric_df.columns:
            plt.figure()
            sns.histplot(self.df[col].dropna(), kde=True, bins=30)
            plt.title(f'Distribution of {col}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'hist_{col}.png'))
            plt.close()

        return self.analysis_results

if __name__=='__main__':
    analyzer = DataAnalyzer("Network_Data/covtype.csv")
    results = analyzer.analyze(target_column="attack_cat", output_dir="eda_output")
