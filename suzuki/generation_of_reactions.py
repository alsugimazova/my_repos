from chython import smiles
from chython.reactor import reactions
reagents_path = 'example.txt'
product = []
counter = 0
with open(reagents_path) as f:
    list_of_molecules = f.readlines()
for mol in list_of_molecules:
    mol = smiles(mol)
    reagent_1 = mol
    counter += 1
    for molec in list_of_molecules[counter:]:
        molec  = smiles(molec)
        reagent_2 = molec
        try:
            reaсtion = next(reactions.suzuki(reagent_1,reagent_2))
        except:
            continue
        for molecule in reaсtion.products:
            molecule = str(molecule)
            product.append(molecule)
with open('product.txt','w') as f:
    for mol in product:
        f.write(f'{mol}\n')
