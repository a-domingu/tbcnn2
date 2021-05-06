
# Step 1: Python 3.8.x and pip3 Install in Ubuntu 20.04 LTS:

sudo apt update

sudo apt -y upgrade

sudo apt install python3-pip

# Control version of Python with:
# python3 -V

# Control pip3 installed with: 
# pip3 --version



# Step 2: Upload zip with project named "tbcnn2.zip" into home directory

sudo apt install unzip

mkdir tbcnn2
mv ./tbcnn2.zip ./tbcnn2/tbcnn2.zip
cd tbcnn2
unzip ./tbcnn2.zip 
rm tbcnn2.zip


# Step 3: Install pip dependencies for project:

pip3 install -r requirements.txt

# Step 4: Add dependencies to path

export PATH=/home/igallegosagastume/.local/bin:$PATH

# Control value of PATH variable:
# echo $PATH

# Step 5: Run the param tester application:

python3 param_tester.py

# Step 6: (Optionally) Run the main application:
# python3 main.py


