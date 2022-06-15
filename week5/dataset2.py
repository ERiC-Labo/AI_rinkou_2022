class Dataset:
    def __init__(self, df):
        self.df = df
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
#         print(type(self.X.iloc[idx,:]))
#         print(type(self.Y.iloc[idx]))
        return self.X.iloc[idx,:].values, self.Y.iloc[idx]