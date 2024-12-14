import pandas as pd
from typing import List

from NaiveBayes import NaiveBayes


if __name__ == '__main__':
    # df_sbytes_bin = create_bins(data, 'sbytes')
    # print(df_sbytes_bin.value_counts())
    # data_test = data.iloc[1]
    # test_sbytes = data_test['sbytes']
    # print(data_test['sbytes'])
    # test_bin = pd.cut([test_sbytes], bins=df_sbytes_bin.cat.categories)[0]
    # print(pd.cut([test_sbytes], bins=df_sbytes_bin.cat.categories)[0])

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
    nb: NaiveBayes = NaiveBayes.load_model()
    for col in nb.data_train.columns:
        print(col)
    print(df.iloc[96])
    print(nb.predict(df.iloc[96]))