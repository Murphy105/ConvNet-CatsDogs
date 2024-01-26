# Création d'un nouveau dossier "cats_vs_dogs_small" qui contient 3 sous-dossiers: "train", "validation", "test".
# Le dossier "train" représente les données d'entrainement et contient les 1000 premières images de chiens et 1000 premières images de chat (dossier chien + dossier chat)
# Le dossier "validation" représente les données de validation et contient les 500 images suivantes de chiens et 500 images suivantes de chat (dossier chien + dossier chat)
# Le dossier "test" représente les données de test et contient les 500 images suivantes de chiens et 500 images suivantes de chat (dossier chien + dossier chat)

# Arboressence initiale                    Arborescence finale:
#  -> cats_vs_dogs_small
#      -> train                           #      -> train
#      -> test1                           #          -> cat (1000 imgs)
                                          #          -> dog (1000 imgs)
                                          #      -> validation
                                          #          -> cat (500 imgs)
                                          #          -> dog (500 imgs)
                                          #      -> test
                                          #          -> cat (500 imgs)
                                          #          -> dog (500 imgs)

import os, shutil, pathlib

# Chemins des datasets
original_dir = pathlib.Path(r"C:\Users\t1307\Documents\MACHINE LEARNING\Deep_Learning_F_Chollet\dogs-vs-cats\train")
new_base_dir = pathlib.Path(r"C:\Users\t1307\Documents\MACHINE LEARNING\Deep_Learning_F_Chollet\cats_vs_dogs_small")

def make_subset(subset_name, start_index, end_index):
    """
    Créé un sous-dossier du dataset final
    - subset_name : nom du sous-dossier
    - start_index : position de la première image dans l'ancien dataset
    - end_index : position de la dernière image dans l'ancien dataset
    
    """
    for category in ("cat", "dog"):
        dir = new_base_dir / subset_name / category
        os.makedirs(dir)
        fnames = [f"{category}.{i}.jpg" for i in range(start_index, end_index)]
        for fname in fnames:
            shutil.copyfile(src=original_dir / fname,
                            dst=dir / fname)

make_subset("train", start_index=0, end_index=1000)
make_subset("validation", start_index=1000, end_index=1500)
make_subset("test", start_index=1500, end_index=2500)
