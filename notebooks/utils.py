# Utils

import pandas as pd
import numpy as np
from dython.nominal import theils_u

def getDuplicateColumns(df):
    ''' Get duplicate columns '''
    duplicateColumnNames = set()
    # Iterate over all the columns in dataframe
    for x in range(df.shape[1]):
        # Select column at xth index.
        col = df.iloc[:, x]
        # Iterate over all the columns in DataFrame from (x+1)th index till end
        for y in range(x + 1, df.shape[1]):
            # Select column at yth index.
            otherCol = df.iloc[:, y]
            # Check if two columns at x 7 y index are equal
            if col.equals(otherCol):
                duplicateColumnNames.add(df.columns.values[y])
    return list(duplicateColumnNames)

# Correlated features list
# get_redundant_pairs works with get_top_abs_correlations
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop


def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

# Correlation between categorical features
# Ref. https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
# Ref. Git https://github.com/shakedzy/dython
def corr_categories(df):
    cols = df.columns
    df1 = pd.DataFrame(columns = ['Var1', 'Var2', 'Corr_Cat'])
    for i in cols:
        #j=i[i+1]
        for j in cols:
            if i != j:
                new_row = {'Var1':i, 'Var2':j, 'Corr_Cat': theils_u(df[i], df[j])} 
                df1 = df1.append(new_row, ignore_index=True)
    return df1.sort_values(by=['Corr_Cat'], ascending=False)


# Interquartiel range outlier removal. USE: rm_IQR_outliers(titanic, 'Fare')
def rm_IQR_outliers(df, var):
    q1 = df[var].quantile(.25)
    q3 = df[var].quantile(.75)
    iqr = q3-q1
    h = 3*iqr # originally 1.5
    df.loc[df[var] > q3+h, var] = q3+h
    df.loc[df[var] < q1-h, var] = q1-h
