def process_df(df):
    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin", "Embarked"], axis=1)
    df["Age"] = df["Age"].fillna(df["Age"].mean())
    df = df.replace("male", 0)
    df = df.replace("female", 1)
    return df

