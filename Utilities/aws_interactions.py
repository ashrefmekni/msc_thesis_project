import boto3
import os
import errno
import streamlit as st

def verify_filetype(file, extension):
    if file.endswith(extension):
        return True
    else:
        return False
    
def create_temp_dir():
    if not os.path.exists("temp"):
        os.makedirs("temp")

def create_temp_file(uploaded_file):
    create_temp_dir()
    temp_location = os.path.join("temp", uploaded_file.name)
    with open(os.path.join("temp", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getvalue())
    print("Temp File created successfully")
    return temp_location

def upload_files(source_path, target_path, all_files, extension):
    s3 = boto3.client('s3')
    BUCKET = "project-thesis"

    for file in all_files:
        print(file)
        if verify_filetype(file.name, extension) == False:
            st.write("You cannot Upload this type of file to this folder: " + target_path)
        else:
            temp_location = create_temp_file(file)            
            target_file = target_path + file.name
            s3.upload_file(temp_location, BUCKET, target_file)
            st.success(file.name + " successfully uploaded to your s3 bucket")

def assert_dir_exists(path):
    """
    Checks if directory tree in path exists. If not it created them.
    :param path: the path to check if it exists
    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def download_files(folder_path):
    """
    Downloads recursively the given S3 path to the target directory.
    :param client: S3 client to use.
    :param bucket: the name of the bucket to download from
    :param path: The S3 directory to download.
    :param target: the local directory to download the files to.
    """
    client = boto3.client('s3')
    bucket = "project-thesis"
    target = os.getcwd() + "/" + folder_path
    # Handle missing / at end of prefix
    if not folder_path.endswith('/'):
        folder_path += '/'

    paginator = client.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=bucket, Prefix=folder_path):
        # Download each file individually
        for key in result['Contents']:
            # Calculate relative path
            rel_path = key['Key'][len(folder_path):]
            # Skip paths ending in /
            if not key['Key'].endswith('/'):
                local_file_path = os.path.join(target, rel_path)
                if os.path.exists(local_file_path):
                    continue
                # Make sure directories exist
                local_file_dir = os.path.dirname(local_file_path)
                assert_dir_exists(local_file_dir)
                client.download_file(bucket, key['Key'], local_file_path)



#download_dir('bucket-name', 'path/to/data', 'downloads')