from keras import Model
from keras.applications.vgg16 import VGG16

model = VGG16(weights='imagenet', include_top=False, input_shape=(1024, 1024, 3))

for layer in model.layers:
    layer.trainable = False


model.summary()

new_model = Model(inputs=model.input, outputs=model.get_layer('block1_conv2').output)



new_model.summary()




