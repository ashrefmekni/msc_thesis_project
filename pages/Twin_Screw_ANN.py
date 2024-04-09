from operator import index
import streamlit as st
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os
import pickle
from ydata_profiling import ProfileReport

import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from Utilities.styles import Styles

custom_styles = Styles()

if 'start_modelling' not in st.session_state:
    st.session_state.start_modelling = False
if 'mlp_clicked' not in st.session_state:
    st.session_state.mlp_clicked = False
if 'decision_clicked' not in st.session_state:
    st.session_state.decision_clicked = False
if 'Kneigh_clicked' not in st.session_state:
    st.session_state.Kneigh_clicked = False
if 'mlp_classifier' not in st.session_state:
    st.session_state.mlp_classifier = None
if 'knn_classifier' not in st.session_state:
    st.session_state.knn_classifier = None
if 'dt_classifier' not in st.session_state:
    st.session_state.dt_classifier = None
    
def click_decision_button():
    st.session_state.decision_clicked = True
def click_kneigh_button():
    st.session_state.Kneigh_clicked = True
def click_start_modelling():
    st.session_state.start_modelling = True
def click_mlp_button():
    st.session_state.mlp_clicked = True

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

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("Twin Screw Page")
    choice = st.radio("Navigation", ["Upload","Profiling","Modelling", "Download"])
    #st.info("This project application helps you build and explore your data.")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Profiling": 
    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report()
    st_profile_report(profile_df)

if choice == "Modelling": 
    # Split the data into features (X) and labels (y)
    st.button('Start Modelling', on_click=click_start_modelling)
    if st.session_state.start_modelling == True and not df.empty:
        X = df.drop(["Folder"], axis=1)
        y = df["Folder"]
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
        mlp_accuracy = st.session_state.mlp_classifier.score(X_test, y_test)
        dt_accuracy = st.session_state.dt_classifier.score(X_test, y_test)
        knn_accuracy = st.session_state.knn_classifier.score(X_test, y_test)
        
        #mlp_accuracy = 0.5981538461538461
        #dt_accuracy = 0.7987692307692308
        #knn_accuracy = 0.6276923076923077
        
        text = f"mlp_accuracy:  {mlp_accuracy}<br>dt_accuracy:  {dt_accuracy}<br>knn_accuracy:  {knn_accuracy}"
        st.markdown(f'<div style="{custom_styles.display_card_style}"><div>{text}</div></div>', unsafe_allow_html=True)
        
        plot_accuracy_graph(mlp_accuracy, dt_accuracy, knn_accuracy)
        st.session_state.start_modelling = False
    

if choice == "Download":
    col1, col2, col3 = st.columns(3)
    with col1:
        st.button('Generate MLP Classifier', on_click=click_mlp_button)
        if st.session_state.mlp_clicked:
            # save the iris classification model as a pickle file
            model_pkl_file = "mlp_class.pkl"  

            with open(model_pkl_file, 'wb') as file:  
                pickle.dump(st.session_state.mlp_classifier, file)
            message = f"The MLP Classifier has succesfully been generated"
            st.markdown(f'<div style="{custom_styles.display_card_style}"><div>{message}</div></div>', unsafe_allow_html=True)
            st.session_state.mlp_clicked = False
    
    with col2:       
        st.button('Generate DecTree Classifier', on_click=click_decision_button)
        if st.session_state.decision_clicked:
            # save the iris classification model as a pickle file
            model_pkl_file = "decision_class.pkl"  

            with open(model_pkl_file, 'wb') as file:  
                pickle.dump(st.session_state.dt_classifier, file)
            message = f"The Descision Tree Classifier has succesfully been generated"
            st.markdown(f'<div style="{custom_styles.display_card_style}"><div>{message}</div></div>', unsafe_allow_html=True)
            st.session_state.decision_clicked = False
    
    with col3:
        st.button('Generate KNeigh Classifier', on_click=click_kneigh_button)
        if st.session_state.Kneigh_clicked == True:
            # save the iris classification model as a pickle file
            model_pkl_file = "Kneigh_class.pkl"  
            with open(model_pkl_file, 'wb') as file:  
                pickle.dump(st.session_state.knn_classifier, file)
            message = f"The Kneighbors Classifier has succesfully been generated"
            st.markdown(f'<div style="{custom_styles.display_card_style}"><div>{message}</div></div>', unsafe_allow_html=True)
            st.session_state.Kneigh_clicked = False
            