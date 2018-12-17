# Article-Popularity-Prediction
The goal of this project is to be able to take an online article and be able to predict how popular it will be.
This Neural Network classifier was build using Java Maven on IntelliJ and  there are two ways so can run this program:
## A
I have packaged our program into a “Jar File” named “BackPropagation.jar” which can be found in the folder named “Packaged_Jar_Project”. 
To run this file, simply open command prompt or bash and type “java -jar BackPropagation.jar”. Instructions will display on how to test the NN.
You can either test the already trained network (Object file saved as TrainedNetwork.txt) or you can start from scratch to train and then test. 
Note if you choose to train the network,I have set a recognition threshold of 80% on the train dataset (you can always change this in Backprop code). 
This means the training section will only terminate once the recognition rate reaches 80% on the train data. The last time I trained our network, I got to that threshold after 250 Epochs. 
Once the training is complete, the trained network will be saved as “TrainedNetwork.txt” replacing the old one.
## B
The second method you you can use to run the NN is to open the Neural Network folder using IntelliJ. Once opened, you can test the NN by simply running the main class named “RunProgram.java”. 
Instructions on how to continue will be displayed which is same as above.
