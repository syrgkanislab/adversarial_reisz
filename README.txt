To set up the environment using the requirements.txt (the one here emulates a default Google Colab environment), run the following code:
cd "/home/liuw/Adversarial Riesz Python/"
module load sloan/python/3.11.4
python3 -m pip install virtualenv --user
virtualenv -p python3 venv
source venv/bin/activate
python3 -m pip install -r requirements.txt

If the directory in which virtualenv is installed is not on PATH, add it using:
export PATH=$PATH:<path of directory>
The directory is probably something like "/home/liuw/.local/bin", and you will be told it after (re)installing virtualenv.

To activate the environment, run:
cd "/home/liuw/Adversarial Riesz Python/"
module load sloan/python/3.11.4
source venv/bin/activate