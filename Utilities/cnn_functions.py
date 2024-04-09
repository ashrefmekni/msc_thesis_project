import numpy as np
from keras.callbacks import EarlyStopping
from tensorflow.python.keras import layers, models
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

class DataUtility:

    def ratios_to_categories(self, argument):
        switcher = {
            0:0.891,
            5:1.055,
            10:1.228,
            15:1.482,
            20:1.763,
            25:1.975,
            30:2.302,
            35:2.598,
            40:2.904,
            45:3.222,
            50:3.691,
            75:1.084,
            425:2.931,
            275:2.102
        }
        return switcher.get(argument, "nothing")

    def convert_test_imgs(self, testimages):
        return np.array(testimages) / 255.0

    def convert_to_np_arrays(self, train_images, train_labels, val_images, val_labels):

        train_images = np.array(train_images)
        train_labels = np.array(train_labels)
        val_images   = np.array(val_images)
        val_labels   = np.array(val_labels)

        print(train_images.shape)
        print(val_images.shape)
        
        return train_images, train_labels, val_images, val_labels
    
    def calculate_scores(self, labels, pred_model):
        mae = mean_absolute_error(labels, pred_model)
        mse = mean_squared_error(labels, pred_model)
        rmse = np.sqrt(mse)
        r2 = r2_score(labels, pred_model)
        
        return mae, mse, rmse, r2


class ModelUtility:

    def create_model(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(91, 53, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1, activation='relu'))

        #model.summary()
        return model#, summary


    def train_cnn_model(self, model, train_images, train_labels, val_images, val_labels, epochs=100):
        early_stop = EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='auto')
        model.compile(optimizer='adam', loss='mse')
        history = model.fit(train_images, train_labels, epochs=epochs,validation_data=(val_images, val_labels), callbacks=[early_stop])
        return history
    
#class Styles:
    
