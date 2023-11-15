from rdkit import Chem
import rdkit.Chem.Lipinski as Lipinksy
import pandas as pd
from tqdm import tqdm

def read(input_path):
    with open(input_path) as f:
        list_of_smiles = [line.rstrip() for line in f]
    return(list_of_smiles)

def main(input_path, output_path):
    dataset = read(input_path)
    molecules = []
    rule_1 = []
    rule_2 = []
    rule_3 = []
    rule_4 = []
    for mol in tqdm(dataset):
        mol = Chem.MolFromSmiles(mol)
        if mol: 
            molecules.append(Chem.MolToSmiles(mol))
            rule_1.append(Lipinksy.NumHDonors(mol))
            rule_2.append(Lipinksy.NumHAcceptors(mol))
            rule_3.append(Lipinksy.rdMolDescriptors.CalcExactMolWt(mol))
            rule_4.append(Lipinksy.rdMolDescriptors.CalcCrippenDescriptors(mol)[0])
    mol_props = {'smiles': molecules,
            'number of Hydrogen Bond Donors': rule_1,
            'number of Hydrogen Bond Acceptors': rule_2,
            'molecular weight' : rule_3,
            'log P': rule_4}       
    lipinsky = pd.DataFrame(mol_props)
    lipinsky.to_csv(output_path, index = False)
    lipinsky.hist(column='number of Hydrogen Bond Donors', color = 'black')
    lipinsky.hist(column='number of Hydrogen Bond Acceptors',color = 'pink')
    lipinsky.hist(column='molecular weight', color = 'grey')
    lipinsky.hist(column='log P', color = 'green')

if __name__ == '__main__':
    main('products.txt','lipinski_properties.csv')