chmod 755 setup_external_libs.sh
chmod 755 requirements.txt
sudo apt update
#sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev
#wget https://www.python.org/ftp/python/3.9.1/Python-3.9.1.tgz
#tar -xf Python-3.9.1.tgz
#cd Python-3.9.1
#./configure --enable-optimizations
#make -j 12
#sudo make altinstall
#python3.9 --version
sudo apt install python3-pip
sudo apt install graphviz
pip install -r requirements.txt