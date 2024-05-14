import numpy as np
import streamlit as st
from tensorflow.python.keras import layers, models
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

from keras.callbacks import EarlyStopping

import sys
import io

import keras
import matplotlib.image as mpimg
from keras import layers, models, Model, Input
from keras.utils import plot_model
import matplotlib.pyplot as plt


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

    def plot_test_graph(self, pred_test, test_images_labels):
        fig = plt.figure()
        # Create a scatter plot for predicted values
        plt.scatter(range(len(pred_test)), pred_test, label='Predicted Viscosity Values')
        plt.plot(range(len(test_images_labels)), test_images_labels, c='red', label='Actual Viscosity Values')
        plt.xlabel('Sample Index')
        plt.ylabel('Viscosity Values')
        plt.title('Predicted vs. Actual Values')

        plt.legend()
        #plt.show()
        st.pyplot(fig)

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
    
    def train_model(self, train_images, train_labels, val_images, val_labels, epochs_number):
        st.header("Model creation")
        DataUtil = DataUtility()
        input_shape = (91, 53, 3)
        
        model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation="relu"),
        ]
    )


        # Redirect stdout to capture model summary
        buffer = io.StringIO()
        sys.stdout = buffer

        # Print model summary
        model.summary()

        # Reset stdout
        sys.stdout = sys.__stdout__

        # Get the model summary from the buffer
        model_summary = buffer.getvalue()

        # Display the model summary
        st.text_area("Model Summary", value=model_summary, height=600)


        early_stop = EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='auto')
        model.compile(optimizer='adam', loss='mse')
        st.header("Model Training")
        history = ""
        with st.spinner("Training On Going"):
            history = model.fit(train_images, train_labels, epochs=epochs_number,validation_data=(val_images, val_labels), callbacks=[early_stop])
            
        if history != "":
            fig = plt.figure()
            plt.plot(history.history['loss'], label='Loss')
            plt.plot(history.history['val_loss'], label = 'Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('MSE Loss')
            plt.ylim([0, 1])
            plt.legend(loc='lower right')
            
            st.pyplot(fig)

            st.header("Measurements & Scores")
            pred_train= model.predict(train_images)
            scores = model.evaluate(train_images, train_labels, verbose=0)
            st.write('Error on training data:')
            st.info(scores)
            pred_valid= model.predict(val_images)
            scores1 = model.evaluate(val_images, val_labels, verbose=0)
            st.write('Error on validation data:')
            st.info(scores1)
            
            mae, mse, rmse, r2 = DataUtil.calculate_scores(train_labels, pred_train)
            st.subheader("Training Error Measurements:")
            st.write('Error MAE:')
            st.info(mae)
            st.write('Error MSE:')
            st.info(mse)
            st.write('Root Mean Squared Error:')
            st.info(rmse)
            st.write('Coefficient of determination ($R^2$):')
            st.info(r2)
            
            mae, mse, rmse, r2 = DataUtil.calculate_scores(val_labels, pred_valid)
            st.subheader("Validation Error Measurements:")
            st.write('Error MAE:')
            st.info(mae)
            st.write('Error MSE:')
            st.info(mse)
            st.write('Root Mean Squared Error:')
            st.info(rmse)
            st.write('Coefficient of determination ($R^2$):')
            st.info(r2)
            
            st.header("Model Plot")
            plot_model(model, to_file='model_plot_2.png', show_shapes=True, show_layer_names=True)

            st.image('model_plot_2.png', use_column_width=True)
            return model


#class Styles:
    
"""
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3,3), activation='relu')(x))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(1, activation='relu'))
"""