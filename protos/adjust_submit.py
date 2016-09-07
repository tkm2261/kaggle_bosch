import pandas


df = pandas.read_csv('submit.csv')

df['Response'] = df['proba'].apply(lambda x: 1 if x > 0.6 else 0)

df[['Id', 'Response']].to_csv('submit2.csv', index=False)

