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
    files = st.file_uploader("Please upload your images", accept_multiple_files = True)

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
        if files:
            if upload_path:
                upload_files(os.getcwd(), upload_path, files, extension)
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
    all_categories = []
    category_images_dict = dict()
    st.header("Create Dataset")
    selected_category = st.selectbox(
        'Which categories you would like to use?',
        ('5.7 rpm', '6.1 rpm', '6.5 rpm', 'all the above'),
        index=None,
        placeholder="Select the categories..."
    )
    if selected_category == 'all the above':
        all_categories = ['5.7 rpm', '6.1 rpm', '6.5 rpm']
    elif selected_category != None:
        all_categories.append(selected_category)
    
    st.write("if you wish to add extra categories, write down the rpm value in this float format x.y rpm")
    new_category = st.text_input('New Category', '0.0 rpm')
    if st.button("add"):
        all_categories.append(str(new_category))
    
    folder_names = setup_folder_categories(all_categories)
    if len(all_categories) > 0:
        st.write("upload the images belonging to each category:")
        for category in all_categories:
            categ_label = all_categories.index(category)
            st.write(category + " →	" + str(categ_label + 1))
            files = st.file_uploader("Please upload your images", key = categ_label, accept_multiple_files = True, type=['png', 'jpg', 'jpeg'])
            category_images_dict[category] = files

    if st.button("Start"):
        with st.spinner("Dataset generation On Going"):
            prepare_dataset(folder_names, category_images_dict)
            st.session_state.dataset_ready = True
    
    if st.session_state.dataset_ready == True:
        df = load_df_from_excel()
        st.dataframe(df)
        st.balloons()