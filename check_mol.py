import pandas 
from rdkit import Chem
from rdkit.Chem.rdmolfiles import SmilesWriter 
df = pandas.read_csv('chemblsmall.csv', sep = ';')
smol = df['Smiles'].to_list()
molecules = []
for mol in smol:
  try:
    mol = Chem.MolFromSmiles(mol)
    Chem.SanitizeMol(mol)
  except TypeError:
    continue
  if mol:
    molecules.append(mol)
with SmilesWriter('chembl.txt') as writer:
  for mol in molecules:
    writer.write(mol)