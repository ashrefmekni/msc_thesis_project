import streamlit as st

from itertools import groupby, cycle

import tensorflow as tf
from sklearn.model_selection import train_test_split

import matplotlib.image as mpimg

from Utilities.cnn_functions import DataUtility, ModelUtility
from Utilities.styles import Styles

DataUtil = DataUtility()
ModelUtil = ModelUtility()
custom_styles = Styles()

with st.sidebar: 
    st.image("assets/head.png")
    st.title("Viscosity Page")
    choice = st.radio("Navigation", ["Create Model","Upload & Test Model"])
    st.info("Select Upload & Test if you have an already existing model")
    
if 'run_clicked' not in st.session_state:
    st.session_state.run_clicked = False
if 'save_clicked' not in st.session_state:
    st.session_state.save_clicked = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'saved_model' not in st.session_state:
    st.session_state.saved_model = None
if 'test_clicked' not in st.session_state:
    st.session_state.test_clicked = False


def click_run_button():
    st.session_state.run_clicked = True
    st.session_state.model_trained = True 
def click_save_button():
    st.session_state.save_clicked = True

def start_from_scratch():
    st.title("CNN Viscosity Model")
    #st.write("Upload your images to train the model")
    with st.expander("Upload your images to train the model"):
        pictures = st.file_uploader("Please upload an image", accept_multiple_files = True, type=['jpg'])
        picture_names = [item.name for item in pictures]
    warning_message = st.empty()
    if pictures:
        warning_message.empty()
        # Split the data into features (X) and labels (y)
        train_size = st.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
        st.write(f"Training Size:")
        st.info(str(train_size) + "%")
        st.write(f"Validation Size:")
        st.info(str(100 - train_size) + "%")
        st.text("")
        validation_size = round(1 - (train_size * 0.01), 2)
        #if st.button('Start Modelling'):
        # Split into training and validation sets
        training_files, validation_files = train_test_split(picture_names, test_size=validation_size, random_state=42)  
        #Data size for training
        train_images = []
        train_labels = []
        val_images   = []
        val_labels   = []
        #print(f"training ratio {training_files}")
        for trainingf_name in training_files:
            trainingratio = trainingf_name.split('-')[0]
            related_pic = list(filter(lambda pic: pic.name == trainingf_name, pictures))[0]
            #print(f"training ratio {trainingratio}")
            train_images.append(mpimg.imread(related_pic))
            train_labels.append(DataUtil.ratios_to_categories(int(trainingratio)))

        for validationf_name in validation_files:
            validationratio = validationf_name.split('-')[0]
            related_pic = list(filter(lambda pic: pic.name == validationf_name, pictures))[0]
            #in case validation folder change data_path to validation data path
            val_images.append(mpimg.imread(related_pic))
            val_labels.append(DataUtil.ratios_to_categories(int(validationratio)))
        

        # Show the first 10 images
        displayedImages = train_images[:10] # your images here
        #print(f"training labels {train_labels[:10]}")
        caption = train_labels[:10] # your caption here
        cols = cycle(st.columns(5)) # st.columns here since it is out of beta at the time I'm writing this
        for idx, displayedImage in enumerate(displayedImages):
            next(cols).image(displayedImage, width=100, caption=caption[idx])
        
        text = f"Training data size {len(training_files)}<br>Validation data size {len(validation_files)}"
        st.markdown(f'<div style="{custom_styles.display_card_style}"><div>{text}</div></div>', unsafe_allow_html=True)

        train_images, train_labels, val_images, val_labels = DataUtil.convert_to_np_arrays(train_images, train_labels, val_images, val_labels)
        # Normalize pixel values between 0 and 1
        train_images, val_images = train_images / 255.0, val_images / 255.0

        label_counts = [(key, len(list(group))) for key, group in groupby(sorted(train_labels))]
        text = "Training label occurrences:<br>"
        for label, count in label_counts:
            text += "Viscosity "f'{label}: Number of Sample = {count}<br>'

        label_counts = [(key, len(list(group))) for key, group in groupby(sorted(val_labels))]
        text += "Validation Label occurences:<br>"
        for label, count in label_counts:
            text += "Viscosity "f'{label}: Number of Sample = {count}<br>'
        
        
        st.markdown(f'<div style="{custom_styles.display_card_style}"><div>{text}</div></div>', unsafe_allow_html=True)

        custom_styles.skip_two_lines()

        epochs_number = st.number_input(
            "Set the number of epochs",
            min_value=10,
            max_value=200,
            value=100,
            step=10
        )
        st.button('Generate Model', on_click=click_run_button)
        if st.session_state.run_clicked:
            st.session_state.saved_model = ModelUtil.train_model(train_images, train_labels, val_images, val_labels, epochs_number)
            #print(st.session_state.saved_model.summary())
            st.session_state.saved_model.save('my_model.keras')
            st.success("Model Saved!")
            st.session_state.save_clicked = False   


    else:
        warning_message.warning('You have not uploaded an image yet', icon="⚠️")

def click_test_button():
    st.session_state.test_clicked = True
    

def use_existing_model():
    st.title("Use Existing CNN Viscosity Model")
    #st.write("Upload your images to train the model")
    with st.expander("Upload your model"):
        model = st.file_uploader("Please upload your model", accept_multiple_files = False, type=['keras'])
    warning_message = st.empty()
    if model:
        model = tf.keras.models.load_model(model.name)
        st.header("Test Model")
        warning_message.empty()
        with st.expander("Upload your test images"):
            pictures = st.file_uploader("Please upload test images", accept_multiple_files = True, type=['jpg'])
            test_files = [item.name for item in pictures]
        if pictures:
            testimages = []
            test_images_labels = []
            for testf_name in test_files:
                testratio = testf_name.split('-')[0]
                related_pic = list(filter(lambda pic: pic.name == testf_name, pictures))[0]
                testimages.append(mpimg.imread(related_pic))
                test_images_labels.append(DataUtil.ratios_to_categories(int(testratio)))
            #st.write(test_images_labels)
            from itertools import groupby
            print([len(list(group)) for key, group in groupby(sorted(test_images_labels))])
            st.button('Test Model', on_click=click_test_button)
            if st.session_state.test_clicked:
                # Normalize pixel values between 0 and 1
                testimages = DataUtil.convert_test_imgs(testimages)
                #DataUtil.unique(test_images_labels)
                pred_test= model.predict(testimages)
                error_card = st.container()
                
                # Inside the container, you can add your content
                with error_card:
                    text = ''
                    label_counts = [(key, len(list(group))) for key, group in groupby(sorted(test_images_labels))]
                    for label, count in label_counts:
                        st.markdown(f"""Viscosity :red[{label}] has 🧮 :red[{count}] samples""")
                        #text += "Viscosity "f'{label} has 🧮 {count} samples<br>'
                    #st.markdown(f'<div style="{custom_styles.display_card_style}"><div>{text}</div></div>', unsafe_allow_html=True)
                    # Calculate the evaluation metrics
                    mae, mse, rmse, r2 = DataUtil.calculate_scores(test_images_labels, pred_test)
                    #mae = 0.36338739177561474
                    #mse = 0.17985903348540133
                    #rmse = 0.4240979055423421
                    #r2 = 0.6867842070759225
                    st.write('Error MAE:')
                    st.info(mae)
                    st.write('Error MSE:')
                    st.info(mse)
                    st.write('Root Mean Squared Error:')
                    st.info(rmse)
                    st.write('Coefficient of determination ($R^2$):')
                    st.info(r2)
                    #text = f"MAE:  {mae}<br>MSE:  {mse}<br>RMSE:  {rmse}<br>R2 score:  {r2}"
                    #st.markdown(f'<div style="{custom_styles.display_card_style}"><div style="{custom_styles.header_style}">Testing Error Measurements:</div><div>{text}</div></div>', unsafe_allow_html=True)

                    total=0
                    for i in range(len(pred_test)):
                        PercentError= (test_images_labels[i]-pred_test[i])/(test_images_labels[i])*100
                        total=total+abs(PercentError)
                    val = 100-total/len(pred_test)
                    #val = 78.504295
                    #text = f"Prediction accuracy on Test Set : {val}"
                    #st.markdown(f'<div style="{custom_styles.display_card_style}"><div>{text}</div></div>', unsafe_allow_html=True)
                    st.write('Prediction accuracy on Test Set:')
                    st.info(val)
                custom_styles.skip_two_lines()

                ModelUtil.plot_test_graph(pred_test, test_images_labels)
            
    else:
        warning_message.warning('You have not uploaded an image yet', icon="⚠️")

if choice == "Create Model":
    start_from_scratch()
elif choice == "Upload & Test Model":
    use_existing_model()

