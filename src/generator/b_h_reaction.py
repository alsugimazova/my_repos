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
            reagent_1 = smiles(mols_pair[0])
            reagent_1.canonicalize()
            reagent_2 = smiles(mols_pair[1])
            reagent_2.canonicalize()
            for reaction in reactions.buchwald_hartwig(reagent_1, reagent_2):
                f.write(f'{str(reaction)}\n')
                break

if __name__ == '__main__':
    run('src/generator/test.txt', 'src/generator/BH_pool.txt')

