import streamlit as st

st.write("---")
st.title("Convolutional Neural Network Using PyTorch")

st.write("## Model Definition and Explanation")

st.markdown(
    """
In our Streamlit app, we use a Convolutional Neural Network (CNN) to classify the images contained in the MNIST dataset. Here is a detailed explanation of each layer in our CNN model and its role in the classification process:

### Model Architecture:
1. **Input Layer**:
    - **Shape**: (28, 28, 1)
    - **Function**: This layer serves as the input for our CNN. The input shape corresponds to the dimensions of the MNIST images (28x28 pixels) with a single color channel (grayscale).

2. **Conv2D Layer 1**:
    - **Filters**: 32
    - **Kernel Size**: (5, 5)
    - **Activation**: ReLU
    - **Function**: This convolutional layer applies 32 filters (output channels) to the input image, each of size 5x5 pixels. The ReLU activation function introduces non-linearity, enabling the network to learn complex patterns. The output is a set of feature maps that highlight different aspects/features of the input image, such as edges and textures.

3. **MaxPooling2D Layer 1**:
    - **Pool Size**: (2, 2)
    - **Function**: This pooling layer reduces the spatial dimensions of the feature maps by taking the maximum value in each 2x2 region. This helps in reducing the computational complexity and preventing overfitting.

4. **Conv2D Layer 2**:
    - **Filters**: 64
    - **Kernel Size**: (5, 5)
    - **Activation**: ReLU
    - **Function**: Similar to the first convolutional layer, this layer applies 64 filters of size 5x5 to the feature maps from the previous layer. The increase in layers allows the model to slightly "overfit" the data, with the aim of extracting high-level features. Future layers will ignore the redundant features and reduce overfitting.

5. **MaxPooling2D Layer 2**:
    - **Pool Size**: (2, 2)
    - **Function**: This pooling layer performs another spatial downsampling, further reducing the size of the feature maps while retaining the most important information.

6. **Flatten Layer**:
    - **Function**: This layer flattens the 2D feature maps into a 1D vector, which can be used as input to the fully connected layers.

7. **Linear (Dense) Layer 1**:
    - **Units**: 128
    - **Activation**: ReLU
    - **Function**: This fully connected Linear (dense) layer consists of 128 neurons. It processes the flattened feature vector and learns complex relationships between the features to perform the classification.
            - Here dense means that every input neuron/node is connected to every output node (preserving the size of the vector)

8. **Dropout Layer**:
    - **Rate**: 0.5
    - **Function**: This dropout layer randomly sets 50% of its inputs to zero during training. This helps in preventing overfitting by ensuring that the model does not rely too heavily on any particular neurons.

9. **Dense Layer 2 (Output Layer)**:
    - **Units**: 10
    - **Activation**: Softmax
    - **Function**: This output layer consists of 10 neurons, each corresponding to one of the digit classes (0-9). The softmax activation function outputs a probability distribution over the 10 classes, indicating the model's confidence in each class.

### Summary:
The CNN model processes the MNIST images through a series of convolutional and pooling layers to extract features, which are then fed into fully connected layers for classification. The model learns to recognize patterns and structures in the images, enabling it to accurately classify handwritten digits.
"""
)

st.write("---")
st.write("Model Training and Validation")
st.image(
    "/Users/lukewilsen/Desktop/IEX/IEX_Training/Neural_Networks/screenshots/Screenshot 2024-05-29 at 7.53.46 PM.png",
    caption="Train and Validation Accuracy through Epoch Progression",
)

st.write("Graphs of model performance through epoch progression")
st.image(
    "/Users/lukewilsen/Desktop/IEX/IEX_Training/Neural_Networks/screenshots/Screenshot 2024-05-29 at 7.54.14 PM.png",
    caption="Model Performance",
)
