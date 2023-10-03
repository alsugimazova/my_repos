from chython import smiles
from chython.reactor import reactions
reagents_path_1 = 'reag1.txt'
reagents_path_2 = 'reag2.txt'
with open(reagents_path_1) as f:
    for line in f:
        reagent_1 = line.strip()
        reagent_1 = smiles(reagent_1)
with open(reagents_path_2) as f:
    for line in f:
        reagent_2 = line.strip()
        reagent_2 = smiles(reagent_2)
try: 
    reaсtion = next(reactions.suzuki(reagent_1,reagent_2))
except:
    print('реакция не идет')
for mol in reaсtion.products:
    mol = str(mol)
with open('product.txt','w') as f:
    f.write(mol)
    