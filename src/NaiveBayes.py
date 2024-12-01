import pandas as pd
from typing import List, Dict

class NaiveBayes:
    def __init__(self, data_train: pd.DataFrame, categorical_columns: List[str], numerical_columns: List[str], target_column: str):
        """
        :param data_train: data used to train the model
        :param categorical_columns: list of column name that contains categorical values
        :param numerical_columns: list of column name that contains numerical values
        :param target_column: column dataframe target. assume only one column
        """
        self.data_train: pd.DataFrame = data_train
        self.categorical_columns: List[str] = categorical_columns
        self.numerical_columns: List[str] = numerical_columns
        self.data_train_freq: Dict[str, Dict[str, Dict[str, int]]] = {} # [key: column name, value: [key: target unique values, value: [key: unique value of column, value: frequency of unique value for each column]]]
        self.data_train_prob: Dict[str, Dict[str, Dict[str, float]]] = {} # [key: column name, value: [key: target unique values, value: [key: unique value of column, value: probability of unique value for each column]]]
        self.target_column: str = target_column
        self.target_freq: Dict[str, int] = {} # [key: target unique values, value: frequency of target unique values]
        self.target_prob: Dict[str, float] = {} # [key: target unique values, value: probability of target unique values]


    def calculate_frequency_column(self):
        for col in self.categorical_columns:
            self.data_train_freq[col] = {}
            for target in self.data_train[self.target_column].unique():
                freq = {}
                for value in self.data_train[col].unique():
                    freq[value] = self.data_train[(self.data_train[self.target_column] == target) & (self.data_train[col] == value)].shape[0]
                self.data_train_freq[col][target] = freq
            print(col, "done")

        for col in self.numerical_columns:
            self.data_train_freq[col] = {}
            # Create binned column
            self.data_train[f'{col}_binned'] = pd.qcut(self.data_train[col], q=5, duplicates='drop')

            for target in self.data_train[self.target_column].unique():
                freq = {}
                for value in self.data_train[f'{col}_binned'].unique():
                    freq[str(value)] = self.data_train[(self.data_train[self.target_column] == target) & (
                                self.data_train[f'{col}_binned'] == value)].shape[0]
                self.data_train_freq[col][target] = freq
            print(col, "done")

        for target in self.data_train[self.target_column].unique():
            self.target_freq[target] = self.data_train[self.data_train[self.target_column] == target].shape[0]
        print(self.target_freq)


    """
    Use quantile to create bins for numerical column
    """
    def create_bins_numerical_column(self, col: str, n_bins: int):
        return pd.qcut(self.data_train[col], n_bins, duplicates='drop')


    """
    Calculate probability based on frequency of each column, relative to the target column
    """
    def calculate_probability_column(self):
        for col in self.categorical_columns:
            self.data_train_prob[col] = {}
            for target in self.data_train[self.target_column].unique():
                prob = {}
                for value in self.data_train[col].unique():
                    prob[value] = self.data_train_freq[col][target][value] / self.data_train.shape[0]
                self.data_train_prob[col][target] = prob

        for col in self.numerical_columns:
            self.data_train_prob[col] = {}
            for target in self.data_train[self.target_column].unique():
                prob = {}
                for value in self.data_train[f'{col}_binned'].unique():
                    prob[str(value)] = self.data_train_freq[col][target][str(value)] / self.data_train.shape[0]
                self.data_train_prob[col][target] = prob

        for target in self.data_train[self.target_column].unique():
            self.target_prob[target] = self.target_freq[target] / self.data_train.shape[0]

    """
    Train the model
    """
    def train(self):
        self.calculate_frequency_column()
        self.calculate_probability_column()


    """
    Predict the data. Input is just one row
    """
    def predict(self, data_test: pd.DataFrame):
        result = {}
        for target in self.data_train[self.target_column].unique():
            prob = self.target_prob[target]
            for col in self.categorical_columns:
                prob *= self.data_train_prob[col][target][data_test[col]]
            for col in self.numerical_columns:
                prob *= self.data_train_prob[col][target][str(data_test[f'{col}_binned'])]
            result[target] = prob
        return max(result, key=result.get)


    """
    Save the model to a txt file
    """
    def save_model(self, model_file_name: str = 'naive_bayes_model.txt'):
        file_name: str = model_file_name
        with open(file_name, 'w') as file:
            file.write(str(self.data_train_prob))
            file.write(str(self.target_prob))


if __name__ == "__main__":
    target_col: str = 'attack_cat'
    cat_col_names: List[str] = ['service', 'state']
    num_col_names: List[str] = ['dur', 'sbytes', 'dbytes']
    all_col: List[str] = cat_col_names + num_col_names
    df: pd.DataFrame = pd.read_csv('../dataset/train/basic_features_train.csv', usecols=all_col)
    df_target: pd.DataFrame = pd.read_csv('../dataset/train/labels_train.csv', usecols=[target_col])
    # Combine with target column
    df = pd.concat([df, df_target], axis=1)
    # Drop NaN
    df = df.dropna()
    nb: NaiveBayes = NaiveBayes(df, cat_col_names, num_col_names, target_col)
    nb.train()
    nb.save_model()
    # for col, value in nb.data_train_freq.items():
    #     for target, freq in value.items():
    #         print(col, target)
    print(df.iloc[96])
    print(nb.predict(df.iloc[96]))
