import pandas


df = pandas.read_csv('submit.csv')

<<<<<<< HEAD
df['proba'] = df['m1']  # df[['m0', 'm1', 'm2']].mean(axis=1)
df['Response'] = df['proba'].apply(lambda x: 1 if x >= 0.24 else 0)
=======
#df['proba'] = df['m2']  # df[['m0', 'm1', 'm2']].mean(axis=1)
df['Response'] = df['proba'].apply(lambda x: 1 if x >= 0.25 else 0)
>>>>>>> dab35e60103a27d8e187a546bd1c609cbd48e038

df[['Id', 'Response']].to_csv('submit1.csv', index=False)
