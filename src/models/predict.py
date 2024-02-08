import numpy as np
from rdkit import Chem
import pickle 
from rdkit.Chem import rdFingerprintGenerator

class Predictor:
    def __init__(self, pkl_file, input_path, output_path = 'data/interim/output_data.txt', model = None):
        self.pkl_file = pkl_file
        self.input_path = input_path
        self.output_path = output_path 
        self.model = model
    
    def load_model(self):
        with open(self.pkl_file, 'rb') as file: 
            model = pickle.load(file)
        return model
        
    def gen_prediction(self, fp):
        model = self.load_model()
        prediction = model.predict([fp])
        return prediction
        
    def main(self):
        with open(self.input_path) as f:
            list_of_molecules = [line.rstrip() for line in f]
        with open(self.output_path,'w') as f:
            for mol in list_of_molecules:
                mol = prepare_molecule(mol)
                if mol is not None:
                    fp = gen_fgp(mol)
                    prediction = self.gen_prediction(fp)
                    for i in prediction:
                        i = str(i)
                    f.write(mol + ' ' +  i  + '\n')
                    
def prepare_molecule(mol):
    atoms_list = ['C', 'O', 'N', 'S', 'Na', 'K', 'F', 'Cl', 'Br', 'I']
    mol = Chem.MolFromSmiles(mol)
    if mol:
        mol_formula = [i.GetSymbol() for i in mol.GetAtoms()]
        for atom in mol_formula :
            if atom in atoms_list:
                mol = Chem.MolToSmiles(mol, canonical=True)
                return mol
        return None
    return None

def gen_fgp(smiles):
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024)
    smiles = Chem.MolFromSmiles(smiles)
    fp = mfpgen.GetFingerprintAsNumPy(smiles)
    return fp
                
predictor = Predictor('models/RFClassifier.pkl', 'data/raw/test.txt')
predictor.main()
