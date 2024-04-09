import streamlit as st
import os

from Utilities.aws_interactions import upload_files, download_files
from Utilities.ann_functions import prepare_dataset, setup_folder_categories, load_df_from_excel

if 'dataset_ready' not in st.session_state:
    st.session_state.dataset_ready = False

with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("Thesis ML Web App")
    choice = st.radio("Navigation", ["Upload", "Download","Create Dataset"])
    #st.info("This project application helps you build and explore your data.")

if choice == "Upload":
    st.title("Upload Your Images")
    pictures = st.file_uploader("Please upload your images", accept_multiple_files = True)

    upload_path = None
    upload_type = st.selectbox(
        'What would you like to upload?',
        ('Images', 'Models', 'Datasets'),
        index=None,
        placeholder="Select a type..."
    )

    if upload_type == 'Images':
        upload_path = st.selectbox(
            'Where would you like to upload?',
            ('RotationCNN', 'ViscosityCNN')
        )
        
        upload_path = upload_path + "/"
        extension = '.png'

    elif upload_type == "Datasets":
        upload_path = upload_type + "/"
        extension = '.csv'
    
    elif upload_type == "Models":
        upload_path = upload_type + "/"
        extension = '.pkl'

    st.write('You Selected This Type: ', upload_type)
    st.write('The Bucket Folder Path: ', upload_path)
    
    if st.button("Upload"):
        if pictures:
            if upload_path:
                print(os.get)
                upload_files(os.getcwd(), upload_path, pictures, extension)
            else:
                st.warning('The Upload Path is not set yet', icon="⚠️")
        else:
            st.warning('Select Some files first', icon="⚠️")

if choice == "Download": 
    st.title("Download Files Locally")
    folder_type = st.selectbox(
        'Which folder would you like to download?',
        ('RotationCNN', 'ViscosityCNN', 'Datasets', 'Models'),
        index=None,
        placeholder="Select folder..."
    )
    
    st.write('You selected:', folder_type)

    if st.button("Download"):
        if folder_type != None:
            folder_path = folder_type + "/"
            st.write('The Bucket Folder Path: ', folder_path)
            download_files(folder_path)
        else:
            st.warning('Select The Folder First', icon="⚠️")

if choice == "Create Dataset":
    st.header("Create Dataset")
    folder_names = setup_folder_categories()
    st.write(folder_names)
    if st.button("Start"):
        with st.spinner("Dataset generation On Going"):
            prepare_dataset(folder_names)
            st.session_state.dataset_ready = True
    if st.session_state.dataset_ready == True:
        df = load_df_from_excel()
        st.dataframe(df)