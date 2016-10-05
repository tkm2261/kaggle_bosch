import pandas


df = pandas.read_csv('submit.csv')

df['proba'] = df['m1']  # df[['m0', 'm1', 'm2']].mean(axis=1)
df['Response'] = df['proba'].apply(lambda x: 1 if x >= 0.24 else 0)

df[['Id', 'Response']].to_csv('submit1.csv', index=False)
