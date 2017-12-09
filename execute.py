from models.basemodel import basemodel
from models.UserCF import userCF
from models.NMF_model import NMF_model
import pandas as pd
import profile

data_df = pd.read_csv("Data/train_data.txt", sep='\t', header=-1)
test_df = pd.read_csv("Data/test_data.txt", sep='\t', header=-1)
model = NMF_model(data_df, 5)
model.train()

model.evaluation(test_df)
