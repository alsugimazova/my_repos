from chython import smiles
from chython.reactor import reactions
from itertools import product
with open('reag1.txt') as f:
    for line in f:
        a = line.strip()
        a = smiles(a)
with open('reag2.txt') as f:
    for line in f:
        b = line.strip()
        b = smiles(b)
try: 
    prod = str(next(reactions.suzuki(a, b)))
    product = prod.split(">>")
    with open('product.txt','w') as f:
        f.write(product[1])
except:
    print('реакция не идет')