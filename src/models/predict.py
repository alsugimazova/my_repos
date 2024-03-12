import numpy as np
from rdkit import Chem
import pickle 
from rdkit.Chem import rdFingerprintGenerator
import click

class Predictor:
    def __init__(self, path_to_model, input_path, output_path, model = None):
        self.path_to_model = path_to_model
        self.input_path = input_path
        self.output_path = output_path 
        self.model = model

    @staticmethod
    def prepare_molecule(mol):
        atoms_list = ['C', 'O', 'N', 'S', 'Na', 'K', 'F', 'Cl', 'Br', 'I']
        mol = Chem.MolFromSmiles(mol)
        if mol:
            mol_formula = set([i.GetSymbol() for i in mol.GetAtoms()])
            for atom in mol_formula :
                if atom in atoms_list:
                    mol = Chem.MolToSmiles(mol, canonical=True)
                    return mol
            return None
        return None
    
    @staticmethod
    def gen_fgp(smiles):
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024)
        smiles = Chem.MolFromSmiles(smiles)
        fp = mfpgen.GetFingerprintAsNumPy(smiles)
        return fp
    
    def load_model(self):
        with open(self.path_to_model, 'rb') as file: 
            self.model = pickle.load(file)
        
    def gen_prediction(self, fp):
        prediction = self.model.predict([fp])
        return str(prediction)[1]
        
    def main(self):
        self.load_model()
        with open(self.input_path) as f:
            list_of_molecules = []
            for line in f:
                str = line.split('>>')
                list_of_molecules.append(str[1])

        with open(self.output_path,'w') as f:
            for mol in list_of_molecules:
                mol = self.prepare_molecule(mol)
                if mol is not None:
                    fp = self.gen_fgp(mol)
                    prediction = self.gen_prediction(fp)
                    f.write(mol + ' ' +  prediction + '\n')
                    
@click.command()
@click.argument('path_to_model', type=click.Path(exists=True))
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path')
def run(path_to_model, input_path, output_path):
    '''Prediction the inhibitory activity of molecules against JAK2'''
    predictor = Predictor(path_to_model, input_path, output_path)
    predictor.main()

if __name__ == '__main__':
    run()
