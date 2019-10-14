import pandas as pd


def load_comparisonXL(method, evaluate="train", dropna=True):
    """Load comparison table."""
    if evaluate == "test":
        e = "['Test']"
    elif evaluate == "in bag":
        e = "['In Bag']"
    elif evaluate == "out of bag":
        e = "['Out of Bag']"
    else:
        e = "['Train']"

    # Import methods
    table = []
    for i in method:
        table.append(pd.read_excel(i + ".xlsx"))

    # Concatenate table
    df = pd.DataFrame()
    for i in range(len(table)):
        df = pd.concat([df, table[i].loc[table[i]['evaluate'] == e].T.squeeze()], axis=1, sort=False)
    df = df.T.drop(columns="evaluate")

    # Remove [ ] from string
    for i in range(len(df)):
        for j in range(len(df.T)):
            if type(df.iloc[i, j]) is str:
                df.iloc[i, j] = df.iloc[i, j][2: -2]

    # Reset index and add methods column
    method_name = []
    for i in range(len(method)):
        name_i = method[i].rsplit('/', 1)[1]
        method_name.append(name_i)
    df = df.reset_index()
    df = pd.concat([pd.Series(method_name, name="method"), df], axis=1, sort=False)
    df = df.drop("index", 1)
    #df = df.set_index("method")

    # drop columns with just nans
    if dropna is True:
        df = df.dropna(axis=1, how='all')

    return df
