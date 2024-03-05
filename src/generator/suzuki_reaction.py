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
            rct = str()
            reagent_1 = mols_pair[0]
            reagent_1 = Chem.MolToSmiles(Chem.MolFromSmiles(reagent_1))
            reagent_1 = smiles(reagent_1)
            rct += str(reagent_1) + '.'
            reagent_1.canonicalize()
            reagent_2 = mols_pair[1]
            reagent_2 = Chem.MolToSmiles(Chem.MolFromSmiles(reagent_2))
            reagent_2 = smiles(reagent_2)
            rct += str(reagent_2) + '>>'
            reagent_2.canonicalize()
            for reaction in reactions.suzuki_miyaura(reagent_1, reagent_2):
                rct += str(reaction.products[0])
                f.write(f'{rct}\n')
                break

if __name__ == '__main__':
    run('src/generator/test.txt', 'src/generator/SM_pool.txt')
    