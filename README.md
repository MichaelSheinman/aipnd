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

* Building the model: This section includes building the model, defining the critertion(I used Adam) and the optimizer.

* Training the model: In this part, the model will go through multiple epochs(4 in my case) and check how far it is from the solution that will classify it correctly. Next the model performs an update by the points it should reach, but it uses a learning rate to avoid going completely to that point. While doing this, I also use the validation set to check the accuracy and the loss on different data

* Testing: This section is also measuring how far the model is from the solution but this time I do not update the weights and I just measure whether the prediction was correct. 

* Saving checkpoint: Now that model is working, I save the checkpoint to be used later. 

* Processing image: For the next couple of parts I had to actually show the images with their most likely parts. To do this, I start by resizing the image, cropping it and normalizing it. 

* Predict: To predict the image classes I run the image through my model and return the classes and the prediction of the images with the greatest probability. 

* Sanity checking: Using the prediction from the predict functions I use matplotlib to plot an inverted bar chart, which consits of the amount of classes return from the predict function with their probabilities. 


## Arguments


In part 2 of the project, the train.py and the predict.py files using argparse to allow the user to specify arguments of their own choice. This arguments include: 

Common: 
* gpu: The argument is a bool, stores True if provided but otherwise is default to False. 



train: 
* data_dir: The directory the data is placed at.  
* arch: pretrained model to be used, increases accuracy. 
* hidden units, learning_rate, epochs: Hyperparameters, specify how the model will train 
* save_dir: checkpoing to save the model once training finished 

predict: 
* image: image to be tested
* top_k: how many classes should be shown 
* checkpoint: The checkpoint to check the image 
* labels: the possible classes the image belongs to


## Credits:
* Partha PratimN: great mentor in the Udacity chat.    
* People in Slank, great community and thanks for everyone for their help :) Details about help I recieved: 
* Saving function: at first I was trying to save the whole model, but a fellow student showed me their way of saving the function and I used it. 
* process_image: I also recieved help with this function when cropping the image. 

