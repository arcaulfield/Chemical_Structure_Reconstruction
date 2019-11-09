import os
import pandas as pd
from src.config import data_path


def clean_mordred_data():
    df1 = pd.read_csv(os.path.join(data_path, "mordred_files", "compound_set1.csv"))
    df2 = pd.read_csv(os.path.join(data_path, "mordred_files", "compound_set2.csv"))
    df3 = pd.read_csv(os.path.join(data_path, "mordred_files", "compound_set3.csv"))
    assert list(df1.columns) == list(df2.columns) == list(df3.columns)
    df = pd.concat([df1, df2, df3], ignore_index=1)

    # drop old indices
    df = df.drop('Unnamed: 0', axis=1)
    df = df.drop('Lipinski', axis=1)
    df = df.drop('GhoseFilter', axis=1)
    # drop rows that contains multiple paths thingy
    df = df.drop(df[df['SpAbs_A'].str.contains('multiple', na=False)].index.tolist())
    df = df.drop(df[df['AATS8dv'].str.contains('invalid', na=False)].index.tolist())
    df = df.drop(df[df['ATSC0c'].str.contains('gasteiger', na=False)].index.tolist())
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(axis=1)
    df = df.dropna(axis=0)

    # extract the key predictors for our dependent variables, leave the rest as explanatory variables
    mordred_x = df.drop(df.columns[df.columns.get_loc('nAromAtom'):df.columns.get_loc('nX') + 1], axis=1)
    mordred_y = df.iloc[:, df.columns.get_loc('nAromAtom'):df.columns.get_loc('nX') + 1]
    return mordred_x.to_numpy()[:-1], mordred_y.to_numpy()[:-1]

