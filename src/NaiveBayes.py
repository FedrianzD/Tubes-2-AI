import joblib
from typing import List, Dict, Union
import pandas as pd
import numpy as np


class NaiveBayes:
    def __init__(self, target_column: str, bin_number: int = 20):
        """
        :param target_column: column dataframe target. assume only one column
        :param bin_number: number of bins for numerical columns
        """
        self.data_train: pd.DataFrame = None
        self.classes_ = None
        self.categorical_columns: List[str] = []
        self.numerical_columns: List[str] = []
        self.data_train_freq: Dict[str, Dict[str, Dict[
            str, int]]] = {}  # [key: column name, value: [key: target unique values, value: [key: unique value of column, value: frequency of unique value for each column]]]
        self.data_train_prob: Dict[str, Dict[str, Dict[
            str, float]]] = {}  # [key: column name, value: [key: target unique values, value: [key: unique value of column, value: probability of unique value for each column]]]
        self.target_column: str = target_column
        self.target_freq: Dict[str, int] = {}  # [key: target unique values, value: frequency of target unique values]
        self.target_prob: Dict[
            str, float] = {}  # [key: target unique values, value: probability of target unique values]
        self.bin_number: int = bin_number
        self.number_of_rows: int = 0
        self.unique_target: np.array = None

    def define_column_type(self):
        """
        Define the type of each column in the dataframe into categorical or numerical
        """
        self.number_of_rows = self.data_train.shape[0]
        for col in self.data_train.columns:
            if col == self.target_column:
                continue
            if self.data_train[col].dtype == 'object' or self.data_train[col].dtype.name == 'category' or \
                    self.data_train[col].dtype.name == 'bool' or self.data_train[col].nunique() < 10 or self.data_train[
                col].dtype == 'str':
                self.categorical_columns.append(col)
            else:
                self.numerical_columns.append(col)

    def print_frequency_details(self):
        """
        Beautify print the frequency details of categorical and numerical columns
        """
        print("\n--- Frequency Details ---")

        # Categorical columns
        for col in self.categorical_columns:
            print(f"\nCategorical Column: {col}")
            for target, freq_dict in self.data_train_freq[col].items():
                print(f"  Target: {target}")
                for value, count in freq_dict.items():
                    print(f"    {value}: {count}")

        # Numerical columns
        for col in self.numerical_columns:
            print(f"\nNumerical Column: {col}")
            for target, freq_dict in self.data_train_freq[col].items():
                print(f"  Target: {target}")
                for value, count in freq_dict.items():
                    print(f"    {value}: {count}")

        # Target frequency
        print("\n--- Target Frequency ---")
        for target, count in self.target_freq.items():
            print(f"{target}: {count}")

    def print_probability_details(self):
        """
        Beautify print the probability details of categorical and numerical columns
        """
        print("\n--- Probability Details ---")

        # Categorical columns
        for col in self.categorical_columns:
            print(f"\nCategorical Column: {col}")
            for target, prob_dict in self.data_train_prob[col].items():
                print(f"  Target: {target}")
                for value, probability in prob_dict.items():
                    print(f"    {value}: {probability:.4f}")

        # Numerical columns
        for col in self.numerical_columns:
            print(f"\nNumerical Column: {col}")
            for target, prob_dict in self.data_train_prob[col].items():
                print(f"  Target: {target}")
                for value, probability in prob_dict.items():
                    print(f"    {value}: {probability:.4f}")

        # Target probabilities
        print("\n--- Target Probabilities ---")
        for target, probability in self.target_prob.items():
            print(f"{target}: {probability:.4f}")

    def calculate_frequency_column(self):
        for col in self.categorical_columns:
            self.data_train_freq[col] = {}
            for target in self.unique_target:
                freq = {}
                for value in self.data_train[col].unique():
                    freq[value] = self.data_train[
                        (self.data_train[self.target_column] == target) & (self.data_train[col] == value)].shape[0]
                self.data_train_freq[col][target] = freq

        for col in self.numerical_columns:
            self.data_train_freq[col] = {}
            # Create binned column
            self.data_train[f'{col}_binned'] = pd.qcut(self.data_train[col], q=self.bin_number, duplicates='drop')

            for target in self.unique_target:
                freq = {}
                for value in self.data_train[f'{col}_binned'].unique():
                    freq[str(value)] = self.data_train[(self.data_train[self.target_column] == target) & (
                            self.data_train[f'{col}_binned'] == value)].shape[0]
                self.data_train_freq[col][target] = freq

        for target in self.unique_target:
            self.target_freq[target] = self.data_train[self.data_train[self.target_column] == target].shape[0]

    def calculate_probability_column(self):
        """
        Calculate probability based on frequency of each column, relative to the target column
        """
        for col in self.categorical_columns:
            self.data_train_prob[col] = {}
            for target in self.unique_target:
                prob = {}
                for value in self.data_train[col].unique():
                    prob[value] = self.data_train_freq[col][target][value] / self.number_of_rows
                self.data_train_prob[col][target] = prob

        for col in self.numerical_columns:
            self.data_train_prob[col] = {}
            for target in self.unique_target:
                prob = {}
                for value in self.data_train[f'{col}_binned'].unique():
                    prob[str(value)] = self.data_train_freq[col][target][str(value)] / self.number_of_rows
                self.data_train_prob[col][target] = prob

        for target in self.unique_target:
            self.target_prob[target] = self.target_freq[target] / self.number_of_rows

    def fit(self, X_train, y):
        """
        Train the model
        :param X_train: training data
        :param y: target of training data
        """
        # Set classes, fix X_train
        self.classes_ = np.unique(y)
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)

        # Fix y datatype
        if isinstance(y, np.ndarray):
            y = pd.Series(y, name=self.target_column)
        elif not isinstance(y, pd.Series):
            raise TypeError("`y` must be a NumPy array or pandas Series")

        # Combine data
        self.data_train = pd.concat([X_train, y], axis=1)
        self.define_column_type()
        self.unique_target = self.data_train[self.target_column].unique()
        self.calculate_frequency_column()
        self.calculate_probability_column()
        return self

    def predict_single(self, data_test: pd.DataFrame):
        """
        Predict the data. Input is just one row.
        """
        result = {}
        for target in self.data_train[self.target_column].unique():
            prob = self.target_prob[target]  # Start with the prior probability of the target class

            for col in self.categorical_columns:
                column_probabilities = self.data_train_prob[col][target]
                value = data_test[col]

                prob *= column_probabilities.get(value, 1e-6)

            for col in self.numerical_columns:
                value = data_test[col]
                column_probabilities = self.data_train_prob[col][target]

                # Find the correct interval
                matching_interval = None
                for interval_str, interval_prob in column_probabilities.items():
                    if interval_str == 'nan':
                        continue

                    # Parse interval string
                    interval_str_num = interval_str[1:-1]
                    left, right = map(float, interval_str_num.split(','))

                    # Check if value fits in this interval
                    if left < value <= right:
                        matching_interval = interval_str
                        prob *= interval_prob
                        break

                # Use a fallback probability if no matching interval is found
                if matching_interval is None:
                    prob *= min(column_probabilities.values(), default=1e-6)

            result[target] = prob

        # Return the class with the highest probability
        return max(result, key=result.get)

    def predict(self, data_test: Union[pd.DataFrame, np.array]):
        """
        Predict the data. Input can be a single row or multiple rows.
        :param data_test: data to predict
        :return: numpy array of predicted values
        """
        if not isinstance(data_test, pd.DataFrame):
            data_test = pd.DataFrame(data_test)
        if len(data_test.shape) == 1 or data_test.shape[0] == 1:
            # If single row
            return np.array([self.predict_single(data_test)])
        else:
            # For multiple rows, iterate for each row
            return np.array([self.predict_single(row) for _, row in data_test.iterrows()])

    def save_model(self, filename: str = 'nb_model.pkl'):
        """
        Save the model to a file
        """
        joblib.dump(self, filename)
        print(f"Model saved as {filename}")

    @staticmethod
    def load_model(filename: str = 'nb_model.pkl'):
        """
        Load the model from a file
        """
        return joblib.load(filename)