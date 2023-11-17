import pandas 
from rdkit import Chem
df = pandas.read_csv('lipinski_properties_no_dubl.csv')
rule_1 = df['number of Hydrogen Bond Donors'].to_list()
rule_2 = df['number of Hydrogen Bond Acceptors'].to_list()
rule_3 = df['molecular weight'].to_list()
rule_4 = df['log P'].to_list()
def check(i,criterion):
    i = int(i)
    if i <= criterion:
        return True
    else:
        return False
counter = 0
for i in range(len(df)):
    r_1 = check(rule_1[i], 5)
    r_2 = check(rule_2[i], 10)
    r_3 = check(rule_3[i], 500)
    r_4 = check(rule_4[i], 5)
    if r_1 is True and r_2 is True and r_3 is True and r_4 is True:
        counter += 1
print(f'out of {len(df)} molecules {counter} are suitable')
    