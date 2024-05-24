#!/bin/bash
set -x

chmod 755 setup_external_libs.sh
chmod 755 requirements.txt
sudo apt update
sudo apt-get update 

### Install necessary packages
sudo apt install python3-pip
sudo apt install graphviz
sudo apt-get install ffmpeg libsm6 libxext6  -y
pip install -r requirements.txt
echo "All requirements are installed."

###setup AWS configuration
aws_dir="$HOME/.aws"
credentials_file="$aws_dir/credentials"

mkdir -p "$aws_dir"

# Create credentials file
cat <<EOF > "$credentials_file"
[default]
aws_access_key_id = AKIAQTQZJDYY6QM5KDQZ
aws_secret_access_key = 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
EOF

echo "AWS credentials file was created successfully in $aws_dir directory."

#start the web application
git config --global user.email "chrefmekni2@gmail.com"
git commit --allow-empty -m "Start building"
git push origin main
streamlit run /index.py