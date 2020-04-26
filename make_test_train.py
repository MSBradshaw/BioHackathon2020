import pandas as pd
import random

df = pd.read_csv('dataset.csv')

fake = df[df['type'] == 'fake']
real = df[df['type'] == 'real']
real.index = list(range(len(real)))
# extract 50 fake

#extract 500 reals
fake_test_ids = random.sample(range(fake.shape[0]), 50)
fake_train_ids = [i for i in list(range(fake.shape[0])) if i not in fake_test_ids ]
fake_train = fake.loc[fake_train_ids]
fake_test = fake.loc[fake_test_ids]



real_test_ids = random.sample(range(real.shape[0]), 500)
real_train_ids = [i for i in list(range(real.shape[0])) if i not in real_test_ids ]
real_train = real.loc[real_train_ids]
real_test = real.loc[real_test_ids]

train = pd.concat([fake_train,real_train])
test = pd.concat([fake_test,real_test])

train.to_csv('train.csv')
test.to_csv('test.csv')