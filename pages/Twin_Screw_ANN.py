from operator import index
import streamlit as st
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import pickle
import joblib

import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from Utilities.styles import Styles
from Utilities.ann_functions import prepare_train_and_test_xy, display_metrics

custom_styles = Styles()
if 'dataframe' not in st.session_state:
    st.session_state.dataframe = pd.DataFrame()

if 'cleaned_dataframe' not in st.session_state:
    st.session_state.cleaned_dataframe = False

if 'start_modelling' not in st.session_state:
    st.session_state.start_modelling = False

#Save the classifiers
if 'mlp_classifier' not in st.session_state:
    st.session_state.mlp_classifier = None
if 'knn_classifier' not in st.session_state:
    st.session_state.knn_classifier = None
if 'dt_classifier' not in st.session_state:
    st.session_state.dt_classifier = None

#Save their scores
if 'mlp_score' not in st.session_state:
    st.session_state.mlp_score = None
if 'knn_score' not in st.session_state:
    st.session_state.knn_score = None
if 'dt_score' not in st.session_state:
    st.session_state.dt_score = None


def click_start_modelling():
    st.session_state.start_modelling = True

def plot_accuracy_graph(mlp_accuracy, dt_accuracy, knn_accuracy):
    # Sample data (replace this with your actual data)
    classifiers = ['mlp', 'decision tree', 'Kneighbors']
    accuracy = [mlp_accuracy, dt_accuracy, knn_accuracy]
    # Plot accuracy comparison
    fig = plt.figure(figsize=(8, 5))
    plt.bar(classifiers, accuracy, color=['blue', 'green', 'orange'])
    plt.title('Accuracy Comparison')
    plt.xlabel('Classifiers')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    st.text("")
    st.text("")
    st.pyplot(fig)

#if os.path.exists('./dataset.csv'): 
#    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar: 
    st.image("assets/Tervek2.png")
    st.title("Twin Screw Page")
    choice = st.radio("Navigation", ["Upload", "Profiling", "Modelling", "Download", "Load Existing Model"])
    #st.info("This project application helps you build and explore your data.")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        st.session_state.dataframe = pd.read_csv(file, index_col=None)
        #st.session_state.dataframe.to_csv('clean_dataset.csv', index=False)
    
    if not st.session_state.dataframe.empty:
        st.dataframe(st.session_state.dataframe)
        # Column selection for dropping
        st.subheader("Clean Dataframe")
        columns_to_drop = st.multiselect("Select Columns to Drop", st.session_state.dataframe.columns)

        if st.button("Drop Selected Columns"):
            # Drop selected columns from the DataFrame
            st.session_state.dataframe = st.session_state.dataframe.drop(columns=columns_to_drop)
            # Display the updated DataFrame
            st.dataframe(st.session_state.dataframe)
            st.session_state.cleaned_dataframe = True

if choice == "Profiling": 
    st.title("Exploratory Data Analysis")
    profile_df = st.session_state.dataframe.profile_report()
    st_profile_report(profile_df)

if choice == "Modelling": 
    st.title("Classifiers Creation & Training")

    # Split the data into features (X) and labels (y)
    train_size = st.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
    st.write(f"Training Size:")
    st.info(str(train_size) + "%")
    st.write(f"Test Size:")
    st.info(str(100 - train_size) + "%")
    st.text("")
    #st.write(round(1 - (train_size * 0.01), 2))
    if st.button('Start Modelling', on_click=click_start_modelling):
        if not st.session_state.dataframe.empty:
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = prepare_train_and_test_xy(st.session_state.dataframe, train_size)

            # Train MLPClassifier
            st.session_state.mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
            st.session_state.mlp_classifier.fit(X_train, y_train)
            # Train KNeighborsClassifier
            st.session_state.knn_classifier = KNeighborsClassifier(n_neighbors=11, p=2)
            st.session_state.knn_classifier.fit(X_train, y_train)
            # Train DecisionTreeClassifier
            st.session_state.dt_classifier = DecisionTreeClassifier(random_state=42)
            st.session_state.dt_classifier.fit(X_train, y_train)
            # Evaluate the models
            st.session_state.mlp_score = st.session_state.mlp_classifier.score(X_test, y_test)
            st.session_state.dt_score = st.session_state.dt_classifier.score(X_test, y_test)
            st.session_state.knn_score = st.session_state.knn_classifier.score(X_test, y_test)
            #
            # Calculate additional metrics
            mlp_y_pred = st.session_state.mlp_classifier.predict(X_test)
            dt_y_pred = st.session_state.dt_classifier.predict(X_test)
            knn_y_pred = st.session_state.knn_classifier.predict(X_test)

            mlp_report = classification_report(y_test, mlp_y_pred, output_dict=True)
            dt_report = classification_report(y_test, dt_y_pred, output_dict=True)
            knn_report = classification_report(y_test, knn_y_pred, output_dict=True)

            mlp_confusion = confusion_matrix(y_test, mlp_y_pred)
            dt_confusion = confusion_matrix(y_test, dt_y_pred)
            knn_confusion = confusion_matrix(y_test, knn_y_pred)

            display_metrics(mlp_report, mlp_confusion, "MLP Classifier Metrics")

            display_metrics(dt_report, dt_confusion, "Decision Tree Classifier Metrics")

            display_metrics(knn_report, knn_confusion, "KNeighbors Classifier Metrics")
    
        else:
            st.warning("Please enter a Dataset to train the ANN model")  
        #mlp_accuracy = 0.5981538461538461
        #dt_accuracy = 0.7987692307692308
        #knn_accuracy = 0.6276923076923077
    if  st.session_state.mlp_score and st.session_state.dt_score and  st.session_state.knn_score:
        #text = f"mlp_accuracy:  { st.session_state.mlp_score}<br>dt_accuracy:  {st.session_state.dt_score}<br>knn_accuracy:  {st.session_state.knn_score}"
        #st.markdown(f'<div style="{custom_styles.display_card_style}"><div>{text}</div></div>', unsafe_allow_html=True)
        st.write('MLP_accuracy:')
        st.info(st.session_state.mlp_score)
        st.write('KNN_accuracy:')
        st.info(st.session_state.knn_score)
        st.write('DTree_accuracy:')
        st.info(st.session_state.dt_score)
        st.subheader("Accuracy Comparison")
        plot_accuracy_graph( st.session_state.mlp_score, st.session_state.dt_score, st.session_state.knn_score)
        st.session_state.start_modelling = False
    
from streamlit_lottie import st_lottie
import requests

@st.cache_data
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url_ship = "https://lottie.host/f2be7250-f072-4a6a-a688-c12357566b00/rco5W5vz9l.json"
lottie_url_robot = "https://lottie.host/778a0be1-4673-4102-8477-05ad740be551/tncwYCu412.json"

def play_animation(animation_url):
    lottie_json = load_lottieurl(animation_url)
    st_lottie(lottie_json, height=400)

if choice == "Download":
    #st.title("Download a Trained Classifier")
    st.subheader("Download a Trained Classifier")
    classifier_type = st.selectbox(
            'Which classifier would you like to download?',
            ('MLP-Classifier', 'DTree-Classifier', 'KNN-Classifier'),
            index=None,
            placeholder="Select classifier..."
        )
    
    if st.button("Download"):
        if classifier_type == "MLP-Classifier":
            model_pkl_file = "mlp_classifier.pkl"
            with open(model_pkl_file, 'wb') as file:  
                pickle.dump(st.session_state.mlp_classifier, file)
            
            play_animation(lottie_url_robot)
            st.success("The MLP Classifier has succesfully been downloaded", icon="✅")

        elif classifier_type == "DTree-Classifier":
            model_pkl_file = "decision_classifier.pkl"  
            with open(model_pkl_file, 'wb') as file:  
                pickle.dump(st.session_state.dt_classifier, file)
            st.success("The Descision Tree Classifier has succesfully been downloaded", icon="✅")
            play_animation(lottie_url_robot)

        elif classifier_type == "KNN-Classifier":
            model_pkl_file = "Kneigh_classifier.pkl"  
            with open(model_pkl_file, 'wb') as file:  
                pickle.dump(st.session_state.knn_classifier, file)
            st.success("The KNN Classifier has succesfully been downloaded", icon="✅")
            play_animation(lottie_url_robot)
    
        else:
            st.warning("Please select a classifier to download", icon="🚨")
 
    #lottie_json_ship = load_lottieurl(lottie_url_ship)
    #st_lottie(lottie_json_ship, height=400)

    if st.session_state.cleaned_dataframe == True:
        st.subheader("Download Cleaned Dataframe")
        if st.button("Download Cleaned Dataframe"):
            play_animation(lottie_url_ship)
            st.session_state.dataframe.to_csv('clean_dataset.csv', index=False)
            st.success("File is successfully downloaded", icon="✅")

if choice == "Load Existing Model":
    st.title("Model Testing")
    model = None
    if st.session_state.dataframe.empty:
        st.warning("Please upload a dataset")
    else:
        X_train, X_test, y_train, y_test = prepare_train_and_test_xy(st.session_state.dataframe, train_size=80)
        use_upload_on = st.toggle("Use an Uploaded Classifier")

        if use_upload_on:
            st.subheader("Upload a trained classifier")
            trained_classifier = st.file_uploader("Upload Your Classifier (.pkl file)")
            if trained_classifier:
                model = joblib.load(trained_classifier.name)
                # Evaluate the model
                accuracy = model.score(X_test, y_test)
                st.subheader("Accuracy")
                st.info(accuracy)
                #st.subheader("Loss")
                #st.info(loss)
                # Calculate additional metrics
                model_prediction = model.predict(X_test)

                model_report = classification_report(y_test, model_prediction, output_dict=True)

                model_confusion = confusion_matrix(y_test, model_prediction)

                display_metrics(model_report, model_confusion, trained_classifier.name)

        else:
            # Select model for testing
            model_option = st.selectbox("Select Model", ["MLP Classifier", "Decision Tree Classifier", "KNeighbors Classifier"])

            if model_option == "MLP Classifier" and st.session_state.mlp_classifier is not None:
                model = st.session_state.mlp_classifier
            elif model_option == "Decision Tree Classifier" and st.session_state.dt_classifier is not None:
                model = st.session_state.dt_classifier
            elif model_option == "KNeighbors Classifier" and st.session_state.knn_classifier is not None:
                model = st.session_state.knn_classifier
            else:
                st.warning("Please train the selected model before testing.")
                st.stop()

            # Evaluate the model
            accuracy = model.evaluate(X_test, y_test)
            st.subheader("Accuracy")
            st.info(accuracy)

            model_prediction = model.predict(X_test)

            model_report = classification_report(y_test, model_prediction, output_dict=True)

            model_confusion = confusion_matrix(y_test, model_prediction)

            display_metrics(model_report, model_confusion, model_option)


    
