# Image classifier application 


## Project parts:

* Notebook: The notebook is the first part of the project. It shows the training, validation and testing process of the data in a Jupyter notebook followed by examples of the images being actually plotted on a bar chart along with their top classes. 

* train.py: The train python file is basically the first part of the jupyter notebook with the addition of command line arguments the user can enter to modify the process. In this part, a network is being created, trained and saved. 

* predict.py: The predict python file is the second part of the project. In this part it is already assumed a checkpoint is created along with a file of labels and using this files the top classes are displayed. 




## Sample output:

* train.py:
training 4 epoch checkpoint, 512 hidden units: https://prnt.sc/kkk78s

* notebook:
http://prntscr.com/kkka01

* predict.py:
10 classes, 1 epoch checkpoint: https://prnt.sc/kkj9il
3 classes, 4 epoch checkpoint: https://prnt.sc/kkkbdl

## Individual parts: 

* Loading the data: Loading the transforms, datasetsets and the dataloaders. This will be done for three separate sets:

** Training set: The set the model iterates through in order to learn over a sequence of epochs. This set is also the only set whose images include special transforms such random rotation and random crop.

** Validation set: The set the model iterates over when checking how each period of training affected the model's accuracy.

** Testing set: A new set the model iterates over when checking the overall accuracy. 

* Building the model: This section includes building the model, defining 


* Creating the model: 
