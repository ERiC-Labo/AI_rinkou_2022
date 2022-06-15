class Dataset:
    def __init__(self, df):
        self.df = df
        self.X = self.df.drop(["Survived"], axis=1)
        self.Y = self.df["Survived"]
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
#         print(type(self.X.iloc[idx,:]))
#         print(type(self.Y.iloc[idx]))
        return self.X.iloc[idx,:].values, self.Y.iloc[idx]