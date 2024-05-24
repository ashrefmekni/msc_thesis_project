import streamlit as st
from st_pages import hide_pages
import os
import time

from Utilities.aws_interactions import upload_files, download_files
from Utilities.ann_functions import prepare_dataset, setup_folder_categories, load_df_from_excel

hide_pages(
    ["index"]
)

if 'dataset_ready' not in st.session_state:
    st.session_state.dataset_ready = False

if 'all_categories' not in st.session_state:
    st.session_state.all_categories = []

if 'new_elements' not in st.session_state:
    st.session_state.new_elements = []


def check_empty_files(categ_dic):
    for key , value in categ_dic.items():
        if value == None or value == []:
            return True, key
    return False, "None"


if st.session_state.user_logged == True:
    #st.set_page_config(initial_sidebar_state="expanded")
    st.empty()
    with st.sidebar: 
        st.image("assets/hand.png")
        st.title("Thesis ML Web Application")
        choice = st.radio("Navigation", ["Upload", "Download","Create Dataset"])
        #st.info("This project application helps you build and explore your data.")

    if choice == "Upload":
        st.title("Upload Your Files")
        files = st.file_uploader("Please upload your images", accept_multiple_files = True)

        upload_path = None
        upload_type = st.selectbox(
            'What would you like to upload?',
            ('Images', 'Models', 'Datasets'),
            index=None,
            placeholder="Select a type..."
        )

        if upload_type == 'Images':
            screw_category = None
            sub_path = None
            upload_path = st.selectbox(
                'Where would you like to upload?',
                ('RotationANN', 'ViscosityCNN')
            )
            if upload_path == 'RotationANN':
               screw_category = st.text_input('What is the screw speed?', '0.0 rpm')
               upload_path = upload_path + "/" + screw_category
            else:
                sub_path = st.selectbox(
                    'To which sub folder would you like to upload?',
                    ('training', 'testing')
                )
                upload_path = upload_path + "/" + sub_path
            extension = '.png'

        elif upload_type == "Datasets":
            upload_path = upload_type + "/"
            extension = '.csv'
        
        elif upload_type == "Models":
            upload_path = upload_type + "/"
            extension = '.pkl'

        st.write('You Selected This Type:')
        st.info(upload_type)
        st.write('The Bucket Folder Path:')
        st.info(upload_path)
        
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
        
        st.write('You selected:')
        st.info(folder_type)

        if st.button("Download"):
            if folder_type != None:
                folder_path = folder_type + "/"
                #st.write('The Bucket Folder Path is:')
                #st.info(folder_path)
                download_files(folder_path)
                st.success("Your download has completed successfully 😊")
            else:
                st.warning('Select The Folder First', icon="⚠️")

    if choice == "Create Dataset":
        selected_options = None
        category_images_dict = dict()
        folder_names = None
        st.header("Create Dataset")
        
        selected_options = st.multiselect(
            'Which categories you would like to use?',
            ['5.7 rpm', '6.1 rpm', '6.5 rpm'],
            placeholder="Select the categories..."
        )
        

        st.session_state.all_categories = selected_options
        st.session_state.all_categories.extend(st.session_state.new_elements)

        st.write("If you wish to add extra categories, write down the rpm value in this float format x.y rpm")
        new_category = st.text_input('New Category', '0.0 rpm')
        if st.button("Add"):
            if str(new_category) not in st.session_state.new_elements:
                st.session_state.new_elements.append(str(new_category))
                st.session_state.all_categories.append(str(new_category))
            st.session_state.add_clicked = False
        
        st.write("Upload the images belonging to each category:")
        for category in st.session_state.all_categories:
            categ_label = st.session_state.all_categories.index(category)
            #st.text(category + " ➡️" + str(categ_label + 1))
            st.markdown("""
                        <style>
                        .big-font {
                            font-size:35px !important;
                            font-weight: bold;
                        }
                        </style>
                        """, unsafe_allow_html=True)

            st.markdown(f'<p class="big-font">{category + " ⤵️"}</p>', unsafe_allow_html=True)
            st.caption("Category Label: " + str(categ_label + 1))
            files = st.file_uploader("Please upload your images", key = categ_label, accept_multiple_files = True, type=['png', 'jpg', 'jpeg'])
            category_images_dict[category] = files


        if st.button("Start"):
            emptiness, missing_category = check_empty_files(category_images_dict)
            if emptiness == False:
                temp_all_categories = st.session_state.all_categories
                #st.write(temp_all_categories)
                folder_names = setup_folder_categories(temp_all_categories)
                #with st.spinner("Dataset generation On Going"):
                #    prepare_dataset(folder_names, category_images_dict)
                #    st.session_state.dataset_ready = True
                with st.status("Generating Dataset...", expanded=True) as status:
                    st.write("Generating Dataset")
                    prepare_dataset(folder_names, category_images_dict)
                    st.session_state.dataset_ready = True
                    time.sleep(2)
                    st.write("Downloading Dataset")
                    time.sleep(1)
                    status.update(label="Dataset Generated and Downloaded!", state="complete", expanded=False)
            
            elif emptiness == True:
                st.warning(f"Please select some pictures for {missing_category} 🚨")

        if st.session_state.dataset_ready == True:
            df = load_df_from_excel()
            st.dataframe(df)
            st.balloons()