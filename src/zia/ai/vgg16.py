import pickle
from pathlib import Path

import cv2
import numpy as np
from keras import Model
from keras.applications.vgg16 import VGG16
from sklearn.ensemble import RandomForestClassifier

from zia.annotations.config.project_config import ResourcePaths
from zia.oven.annotations.workflow_visualizations.util.image_plotting import plot_pic

model = VGG16(weights='imagenet', include_top=False, input_shape=(1024, 1024, 3))

for layer in model.layers:
    layer.trainable = False

model.summary()

new_model = Model(inputs=model.input, outputs=model.get_layer('block1_conv2').output)

new_model.summary()

# loading training data
resource_paths = ResourcePaths("sample_data")

images = []
masks = []

for file in resource_paths.mask_path.iterdir():
    masks.append(cv2.imread(str(file), 0))
    images.append(cv2.imread(str(resource_paths.image_path / file.name)))

x_train = np.array(images[:1])
y_train = np.array(masks[:1])

x_test = np.array(images[4:])
y_test = np.array(images[4:])

features = new_model.predict(x_train)

print(x_train.shape, y_train.shape)

for img in y_train:
    plot_pic(img)

X = features
X = X.reshape(-1, X.shape[3])
Y = y_train.reshape(-1)

model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)

print("Fitting Random Forest Classifier")
model.fit(X, Y)

with open(Path(__file__).parent / "models/RandomForestClassifier.pkl", "wb") as f:
    pickle.dump(model, f)

test_image = x_test

for img in test_image:
    plot_pic(img)

x_test_feature = new_model.predict(test_image)
x_test_feature = x_test_feature.reshape(-1, x_test_feature.shape[3])

predictions = model.predict(x_test_feature)

prediction_images = predictions.reshape(x_test.shape[:3])

for prediction in prediction_images:
    plot_pic(prediction)
