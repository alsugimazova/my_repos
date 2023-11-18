from chython import smiles
from chython.reactor import reactions
from itertools import combinations
from tqdm import tqdm
from rdkit import Chem

def read(reagents_path):
    with open(reagents_path) as f:
        list_of_molecules = [line.rstrip() for line in f]
    return(list_of_molecules)

def run(input_path, output_path):
    dataset = read(input_path)
    with open(output_path,'w') as f:
        for mols_pair in tqdm(combinations(dataset, 2)):
            reagent_1 = mols_pair[0]
            reagent_1 = Chem.MolToSmiles(Chem.MolFromSmiles(reagent_1))
            reagent_1 = smiles(reagent_1)
            reagent_1.canonicalize()
            reagent_2 = mols_pair[1]
            reagent_2 = Chem.MolToSmiles(Chem.MolFromSmiles(reagent_2))
            reagent_2 = smiles(reagent_2)
            reagent_2.canonicalize()
            for reaction in reactions.suzuki(reagent_1, reagent_2):
                f.write(f'{str(reaction.products[0])}\n')
                break
            for reaction in reactions.suzuki_amide(reagent_1, reagent_2):
                f.write(f'{str(reaction.products[0])}\n')
                break

if __name__ == '__main__':
    run('input_path.txt', 'suzuki_products.txt')               
