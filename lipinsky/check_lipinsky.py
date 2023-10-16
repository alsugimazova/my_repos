from rdkit import Chem
import rdkit.Chem.Lipinski as Lipinksy
import pandas as pd
product_path = 'product.txt'
with open(product_path) as f:
        list_of_smiles = [line.rstrip() for line in f]
molecules = []
for mol in list_of_smiles:
    try:
        mol = Chem.MolFromSmiles(mol)
    except TypeError:
        continue
    if mol:
        molecules.append(mol)
rule_1 = []
rule_2 = []
rule_3 = []
rule_4 = []
for mol in molecules:
    rule_1.append(Lipinksy.NumHDonors(mol))
    rule_2.append(Lipinksy.NumHAcceptors(mol))
    rule_3.append(Lipinksy.rdMolDescriptors.CalcExactMolWt(mol))
    rule_4.append(Lipinksy.rdMolDescriptors.CalcCrippenDescriptors(mol)[0])
view = {'smiles': list_of_smiles,
        'number of Hydrogen Bond Donors': rule_1,
        'number of Hydrogen Bond Acceptors': rule_2,
        'molecular weight' : rule_3,
        'log P': rule_4}       
lipinsky = pd.DataFrame(view)
lipinsky.to_csv('lipinsky.csv', index = False)