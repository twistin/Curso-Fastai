from fastai.vision import *

# Importar los modelos y la imagen
path_resnet = Path("C:/Users/mfnunez/Documents/fastai/Curso-Fastai/Clase1_Clasificación")
learn34 = load_learner(path_resnet, 'fastai_01_resnet34.pkl')
learn50 = load_learner(path_resnet, 'fastai_01_resnet50.pkl')
img = open_image(path_resnet/'gato_persa.jpg')

# Predicción de resnet 34
predict_class, predict_idx, outputs = learn34.predict(img)
print("Predicción de resnet34: ", end='')
print(predict_class)

#Predicción de resnet50
predict_class, predict_idx, outputs = learn50.predict(img)
print("Predicción de resnet50: ", end='')
print(predict_class)

