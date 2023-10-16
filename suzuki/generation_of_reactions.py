from chython import smiles
from chython.reactor import reactions
from itertools import combinations
from tqdm import tqdm

def read(reagents_path):
    with open(reagents_path) as f:
        list_of_molecules = [line.rstrip() for line in f]
    return(list_of_molecules)

def run():
    dataset = read('zinc.smi')
    path = 'product_zinc.txt'
    with open(path,'w') as f:
        for mols_pair in tqdm(combinations(dataset, 2)):
            reagent_1 = mols_pair[0]
            reagent_1 = smiles(reagent_1)
            reagent_1 .canonicalize()
            reagent_2 = mols_pair[1]
            reagent_2 = smiles(reagent_2)
            reagent_2 .canonicalize()
            for reaction in reactions.suzuki(reagent_1, reagent_2):
                f.write(f'{str(reaction.products[0])}\n')

if __name__ == '__main__':
    run()
