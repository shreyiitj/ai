import pandas as pd
import numpy as np
from keras.preprocessing import image
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tqdm import tqdm


train_csv_path = '/kaggle/input/data/train.csv'
test_csv_path = '/kaggle/input/data/test.csv'
train_csv = pd.read_csv(train_csv_path)
test_csv = pd.read_csv(test_csv_path)



train_dir_path = '/kaggle/input/data/train/'
train_image = []
for img_name in tqdm(train_csv['image_id']):
    img_path = train_dir_path + str(img_name) + '.jpg'
    img = image.load_img(img_path, target_size=(28,28, 1), grayscale=True)
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)

train_image_x = np.array(train_image)
train_y = train_csv['category'].values
train_y = to_categorical(train_y)

x_train, x_test, y_train, y_test = train_test_split(train_image_x, train_y, random_state=42, test_size=0.2)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.20))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(103, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))

train_dir_path = '/kaggle/input/data/test/'
test_image = []
for img_name in tqdm(test_csv['image_id']):
    img_path = train_dir_path + str(img_name) + '.jpg'
    img = image.load_img(img_path, target_size=(28,28, 1), grayscale=True)
    img = image.img_to_array(img)
    img = img/255
    test_image.append(img)

test_image_x = np.array(test_image)

prediction = model.predict_classes(test_image_x)

sample = pd.read_csv('/kaggle/input/data/sample_submission.csv')
sample['label'] = prediction
sample.to_csv('/kaggle/input/data/final_submission.csv', header=True, index=False)

