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
    for i in list_of_molecules[counter:]:
        check = True
        i  = smiles(i)
        reagent_2 = i
        try:
            reaсtion = next(reactions.suzuki(reagent_1,reagent_2))
        except:
            check = False
        if check == True: 
            for moll in reaсtion.products:
                moll = str(moll)
                product.append(moll)
with open('product.txt','w') as f:
    for mol in product:
        f.write(f'{mol}\n')