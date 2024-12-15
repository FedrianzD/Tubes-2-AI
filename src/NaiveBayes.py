import joblib
from typing import List, Dict, Union
import pandas as pd
import numpy as np


class NaiveBayes():
    def __init__(self, target_column: str, bin_number: int = 20):
        """
        :param data_train: data used to train the model
        :param categorical_columns: list of column name that contains categorical values
        :param numerical_columns: list of column name that contains numerical values
        :param target_column: column dataframe target. assume only one column
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
        self.binning_info: Dict[str, pd.IntervalIndex] = {}  # [key: column name, value: binning information]
        self.bin_number: int = bin_number

    def define_column_type(self):
        """
        Define the type of each column in the dataframe
        """
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
        Beautifully print the frequency details of categorical and numerical columns
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
        Beautifully print the probability details of categorical and numerical columns
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
            for target in self.data_train[self.target_column].unique():
                freq = {}
                for value in self.data_train[col].unique():
                    freq[value] = self.data_train[
                        (self.data_train[self.target_column] == target) & (self.data_train[col] == value)].shape[0]
                self.data_train_freq[col][target] = freq

        for col in self.numerical_columns:
            self.data_train_freq[col] = {}
            # Create binned column
            self.data_train[f'{col}_binned'] = pd.qcut(self.data_train[col], q=self.bin_number, duplicates='drop')

            for target in self.data_train[self.target_column].unique():
                freq = {}
                for value in self.data_train[f'{col}_binned'].unique():
                    freq[str(value)] = self.data_train[(self.data_train[self.target_column] == target) & (
                            self.data_train[f'{col}_binned'] == value)].shape[0]
                self.data_train_freq[col][target] = freq

        for target in self.data_train[self.target_column].unique():
            self.target_freq[target] = self.data_train[self.data_train[self.target_column] == target].shape[0]

    def calculate_probability_column(self):
        """
        Calculate probability based on frequency of each column, relative to the target column
        """
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

    def fit(self, X_train, y):
        """
        Train the model
        """
        # Ensure X is a
        self.classes_ = np.unique(y)
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)

        # Ensure y is a Series
        if isinstance(y, np.ndarray):
            y = pd.Series(y, name=self.target_column)
        elif not isinstance(y, pd.Series):
            raise TypeError("`y` must be a NumPy array or pandas Series")

        # Combine data
        self.data_train = pd.concat([X_train, y], axis=1)
        self.define_column_type()
        print(self.data_train.head())
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

            # Handle categorical columns
            for col in self.categorical_columns:
                column_probabilities = self.data_train_prob[col][target]
                value = data_test[col]

                # Use a fallback probability if the value is missing
                prob *= column_probabilities.get(value, 1e-6)  # Fallback to a small probability (e.g., 1e-6)

            # Handle numerical columns
            for col in self.numerical_columns:
                value = data_test[col]
                column_probabilities = self.data_train_prob[col][target]

                # Find the correct interval
                matching_interval = None
                for interval_str, interval_prob in column_probabilities.items():
                    if interval_str == 'nan':
                        continue  # Skip invalid entries

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
        if not isinstance(data_test, pd.DataFrame):
            data_test = pd.DataFrame(data_test)
        print(data_test.head())
        if len(data_test.shape) == 1 or data_test.shape[0] == 1:
            # If it's a single row (either as a Series or single-row DataFrame)
            return np.array([self.predict_single(data_test)])
        else:
            # For multiple rows, apply predict_single to each row
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