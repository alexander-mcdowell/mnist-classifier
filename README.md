# mnist-classifier
A simple handwritten digit classifier trained using the MNIST dataset. Comes with an application to draw digits and have the application recognize them.

**Architechture**
- Conv2D (32 filters, kernel size 3, stride 2)
- MaxPool2D (pool size 2, stride 2)
- Conv2D (64 filters, kernel size 3, stride 2)
- MaxPool2D (pool size 2, stride 2)
- Conv2D (128 filters, kernel size 3, stride 2)
- MaxPool2D (pool size 2, stride 2)
- Dense (512 neurons, relu activation)
- Dense (256 neurons, relu activation)
- Dense (128 neurons, relu activation)
- Dense (10 neurons, softmax activation)

Accuracy is high for the MNSIT dataset (training, validation, and testing sets alike) but performs poorly during testing with the application. Possibly an issue related to overfitting that I will need to look into in the future.

Modifying the `train` parameter on line 7 allows you to train/test the model. `True` will train and test a new model using the application and `False` will test the model using the drawing application.
