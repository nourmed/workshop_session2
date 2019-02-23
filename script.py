import keras
from keras.preprocessing.image import img_to_array, load_img
from keras.models import load_model
import os
import numpy as np

model_name = 'saved_models/keras_cifar10_trained_model.h5'
img_name = 'test/114.png'
#model_dir= os.path.join(os.getcwd(), 'saved_models')
#model_path = os.path.join(model_dir, model_name)
#img_path= os.path.join(os.getcwd(), 'test/1.png')



model= load_model(model_name)
x_test= load_img(img_name)

x_test_array= img_to_array(x_test)
x_test_reshaped=x_test_array.reshape((1,)+x_test_array.shape)
print(x_test_reshaped.shape)
y= model.predict(x_test_reshaped)
print(y)
c = np.argmax(y)





Switcher={
	0:"Avion",
	1:"Automobile",
	2:"Oiseau",
	3:"Chat",
	4:"Gazelle",
	5:"Chien",
	6:"Grenouille",
	7:"cheval",
	8:"Bateau",
	9:"Camion"
}
print(Switcher.get(c))