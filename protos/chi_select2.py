import pandas

df = pandas.read_csv('train_chi_all_10000.csv.gz')

df_s = df.sum()
idx = df_s[df_s > 10].index.values
if idx.shape[0] == 1:
    print('no col')
    exit()

print(idx.shape)
df[['Id'] + list(idx)].to_csv('train_chi_all_10000_sle.csv.gz', compression='gzip')


df = pandas.read_csv('test_chi_all_10000.csv.gz')
df[['Id'] + list(idx)].to_csv('test_chi_all_10000_sle.csv.gz', compression='gzip')
