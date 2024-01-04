# adversarial_reisz
Adversarial estimation of Reisz representers.

To run the estimation, open Results.ipynb on Google Colaboratory and run all the cells.

Alternatively, you can instead run the estimation outside of Google Colab. Clone this repository, use requirements.txt to create a virtual environment (this mimics the default Google Colab environment), and run Results.ipynb within it. Warning: the performance can sometimes be very poor if not using Google Colab due to an unknown bug involving joblib.Parallel.

(Note: the leftover files AdversarialReisz.ipynb and 401k.ipynb, which are not used for the paper, do not have a fixed seed for advnnet.)
