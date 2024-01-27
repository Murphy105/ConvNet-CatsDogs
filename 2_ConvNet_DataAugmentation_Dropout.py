import matplotlib.pyplot as plt

from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

""" --- Pré-Traitement des données d'entrée --- """
#  Les transformations pour l'augmentation de données sont définies dans l'ImageDataGenerator (UNIQUEMENT pour l'entrainement)
train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=40,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
# Chemin du dataset
dataset_dir = pathlib.Path(r"cats_vs_dogs_small")
# Les images sont redimensionnées à la taille (150x150) et sont générées par lot de 20
train_generator = train_datagen.flow_from_directory(dataset_dir/"train", target_size = (150,150), batch_size = 20, class_mode = 'binary')
validation_generator = validation_datagen.flow_from_directory(dataset_dir/"validation", target_size = (150,150), batch_size = 20, class_mode = "binary")
test_generator = test_datagen.flow_from_directory(dataset_dir/"test", target_size = (150,150), batch_size = 20, class_mode = "binary")

""" --- Création du modèle ConvNet --- """
# Même architecture que le ConvNet précédent
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPool2D((2,2)))
# Ajout d'une couche Dropout
model.add(layers.Dropout(0.5))
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
history = model.fit(train_generator, steps_per_epoch=100, epochs=100, validation_data = validation_generator, validation_steps=50)

""" --- Sauvegarde du modèle à a fin de l'entraînement --- """
model.save('ConvNet2.h5')

""" --- Tracer les courbes d'entrainements --- """
# Voir 1_ConvNet_accuracy.png et 1_ConvNet_loss.png
accuracy = history.history["acc"]
val_accuracy = history.history["val_acc"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, "bo", label="Training accuracy")
plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()

""" --- Evaluation du modèle sur le dataset de test --- """
test_loss, test_acc = model.evaluate(test_generator, steps = 50)
