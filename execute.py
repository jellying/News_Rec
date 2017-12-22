from models.basemodel import basemodel
from models.UserCF import userCF
<<<<<<< HEAD
from models.ItemCB import itemCB
from models.NMF_model import NMF_model
import pandas as pd

# data_df = pd.read_csv("Data/ml_100k/ml_train.txt", sep=',', header=-1)
# test_df = pd.read_csv("Data/ml_100k/ml_test.txt", sep=',', header=-1)
data_df = pd.read_csv("Data/train_data.txt", sep='\t', header=-1)
test_df = pd.read_csv("Data/test_data.txt", sep='\t', header=-1)
# model = NMF_model(data_df, 5)
# model = userCF(data_df, 5)
model = itemCB(data_df, 5)
model.train()
=======
from models.NMF_model import NMF_model
import pandas as pd
import profile

data_df = pd.read_csv("Data/train_data.txt", sep='\t', header=-1)
test_df = pd.read_csv("Data/test_data.txt", sep='\t', header=-1)
model = NMF_model(data_df, 5)
model.train()

>>>>>>> 450b18c77e38ddb767749e30f4ce95135d9b2fce
model.evaluation(test_df)
