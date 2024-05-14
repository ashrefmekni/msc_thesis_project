import streamlit as st
import os
import time

from Utilities.aws_interactions import upload_files, download_files
from Utilities.ann_functions import prepare_dataset, setup_folder_categories, load_df_from_excel

if 'dataset_ready' not in st.session_state:
    st.session_state.dataset_ready = False

if 'all_categories' not in st.session_state:
    st.session_state.all_categories = []


import boto3
from botocore.exceptions import ClientError

# Initialize AWS Cognito client
cognito_client = boto3.client('cognito-idp', region_name='us-east-1')

# Define Streamlit login form
def login():
    st.title('Login')
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')
    if st.button('Login'):
        try:
            # Authenticate user
            response = cognito_client.admin_initiate_auth(
                UserPoolId='us-east-1_63QQlSUm0',
                ClientId='6akm6miipm3qlsp8f7vkjfq9kh',
                AuthFlow='ADMIN_USER_PASSWORD_AUTH',
                AuthParameters={
                    'USERNAME': username,
                    'PASSWORD': password
                }
            )
            st.write(response)
            st.success('Login successful!')
            # Access token can be retrieved from response['AuthenticationResult']['AccessToken']
        except ClientError as e:
            st.write(e)
            st.error('Login failed. Please check your credentials.')


login()

with st.sidebar: 
    st.image("assets/hand.png")
    st.title("Thesis ML Web Application")
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
            st.write('The Bucket Folder Path is: ', folder_path)
            download_files(folder_path)
            st.success("Your download has completed successfully 😊")
        else:
            st.warning('Select The Folder First', icon="⚠️")

if choice == "Create Dataset":
    #all_categories = []
    category_images_dict = dict()
    st.header("Create Dataset")
    st.session_state.all_categories = st.multiselect(
        'Which categories you would like to use?',
        ['5.7 rpm', '6.1 rpm', '6.5 rpm'],
        placeholder="Select the categories..."
    )
    st.write("if you wish to add extra categories, write down the rpm value in this float format x.y rpm")
    new_category = st.text_input('New Category', '0.0 rpm')
    if st.button("add"):
        st.session_state.all_categories.append(str(new_category))
    
    folder_names = setup_folder_categories(st.session_state.all_categories)
    if len(st.session_state.all_categories) > 0:
        st.write("upload the images belonging to each category:")
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
        
    if st.session_state.dataset_ready == True:
        df = load_df_from_excel()
        st.dataframe(df)
        st.balloons()