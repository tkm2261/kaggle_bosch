import pandas


df = pandas.read_csv('submit.csv')

#df['proba'] = df[['m0', 'm1']].max(axis=1)  # df[['m0', 'm1', 'm2']].mean(axis=1)
df['Response'] = df['proba'].apply(lambda x: 1 if x >= 0.358 else 0)
df['Id'] = df['Id'].astype(int)
df[['Id', 'Response']].to_csv('submit1.csv', index=False)
