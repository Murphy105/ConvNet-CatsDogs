from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

""" --- Pré-Traitement des données d'entrée --- """
#  Les données d'entrée sont ramenés à des flottants compris entre 0 et 1
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
# Chemin du dataset
dataset_dir = pathlib.Path(r"cats_vs_dogs_small")
# Les images sont redimensionnées à la taille (150x150) et sont générées par lot de 20
train_generator = train_datagen.flow_from_directory(dataset_dir/"train", target_size = (150,150), batch_size = 20, class_mode = 'binary')
validation_generator = validation_datagen.flow_from_directory(dataset_dir/"validation", target_size=(150,150), batch_size = 20, class_mode = "binary")

""" --- Création du modèle ConvNet --- """
# Modèle construitsen alternant couche de convolution et couche MaxPool
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPool2D((2,2)))
# Couche Flatten pour passer d'un tenseur 3D à un tenseur 1D avant le classifieur
model.add(layers.Flatten())
# Ajout d'un classifieur composé d'une couche Dense(512) et d'une couche Dense(1) pour effectuer la classification binaire "chien ou chat" 
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

""" --- Compilation du modèle --- """
# Fonction de perte "binary_crossentropy" adaptée à la classification binaaire
model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(learning_rate=1e-4),metrics=['acc'])

""" --- Entrainement du modèle --- """
# L'historique de l'entraînement est stocké dans la variable 'history' pour pouvoir visualiser les courbes d'entrapinement à posteriori
history = model.fit(train_generator, steps_per_epoch=100, epochs=30, validation_data = validation_generator, validation_steps=50)

""" --- Sauvegarde du modèle à a fin de l'entraînement --- """
model.save('cats_dogs_small.h5')
