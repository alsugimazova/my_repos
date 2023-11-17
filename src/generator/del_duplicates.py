import pandas 
df = pandas.read_csv('lipinski_properties.csv')
size_1 = df['smiles'].to_list()
print(len(size_1))
df_new = df.drop_duplicates()
df_new.to_csv('lipinski_properties_no_dubl.csv', index = False)
df_2 = pandas.read_csv('lipinski_properties_no_dubl.csv')
size_2 = df_2['smiles'].to_list()
print(len(size_2))