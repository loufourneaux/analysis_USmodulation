## Description

Data analysis script for the US neuromodulation experiments. To be used with the results in the form .adicht 

Run the run.py file to study the results. The plots and statistical results are saved in the folders ANOVA and figures. 

Different files:
- data_loader_3.py: loads the data 
- preprocessing.py: filters, segments, ... the data is ready to be studied 
- processing.py: action potential extraction and features extraction
- statistical_test.py: tests and plots 
- visualization.py: plots which are saved under figures folder


# Install dependencies
pip install -r requirements.txt

( /!\ adi-reader package not available on mac)

Done by Lou Fourneaux, supervised by Elena Vicari
Contact: lou.fourneaux@epfl.ch

