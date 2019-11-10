# Chemical_Structure_Reconstruction
## The Task
We developped an algorithm that can identify the least amount of Mordred descriptors needed to reverse engineer chemical structures. 
## Our Approach
We identifed 17 core structural Mordred descriptors that would allow a researcher to effectively deduce the structure of a molecule. Without these core structural descriptors, a researcher would not be able to accurately sythesize any structure. Our model trains using all of the other Mordred descriptors in order to predict the value of these core structural desciptors. The model is versatile and is able to integrate core structural descriptors as predictors should the user choose to do so. 

## PCA
We first attempted data reduction in order to identify the non-core Mordred descriptors that could explain the most variation in our data (i.e. the descriptors that best differentiate the chemical compounds from each other). This was done via Principal Component Analysis (PCA), a classic data reduction technique.

## The Steps
1. We cleaned the data. We removed any irregularities from it, and ensured that it consisted only of numerical values. 
2. We extracted the 17 core structural descriptors. 
3. We trained a Multiple Output Regressor to predict the 17 core structural descriptors. Using all Mordred descriptors, the model was able to predict of all these descriptors with 100% 5-fold cross validation accuracy. 
4. We iteratively thresholded the data, removing the data with the least variance, in order to find the least amount of descriptors required for the model to accurately predict the 17 core structureal descriptors. 
## Results

## Give it a try! 
In order to run our model, do the following:


