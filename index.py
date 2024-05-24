import streamlit as st
from st_pages import hide_pages
import boto3
from botocore.exceptions import ClientError

if 'user_logged' not in st.session_state:
    st.session_state.user_logged = False

# Initialize AWS Cognito client
cognito_client = boto3.client('cognito-idp', region_name='us-east-1')

# Define Streamlit login form
def login(username, password):
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
        st.success('Login successful!')
        st.session_state.user_logged = True
        st.session_state.hide_form = True
        # Access token can be retrieved from response['AuthenticationResult']['AccessToken']
    except ClientError as e:
        st.write(e)
        st.error('Login failed. Please check your credentials.')


hide_pages(
    ["Twin_Screw_ANN", "Viscosity_CNN", "Homepage"]
)

if st.session_state.user_logged == False:
    loginform = st.container()
    with loginform:
        st.title('Login')
        username = st.text_input('Username')
        password = st.text_input('Password', type='password')
        if st.button('Login'):
            login(username, password)
        
        if st.session_state.user_logged == True:
            st.switch_page("pages/Homepage.py")