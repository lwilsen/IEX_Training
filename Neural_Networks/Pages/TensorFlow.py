import streamlit as st

st.title("Convolutional Neural Network Using Tensorflow")

st.markdown(
    """
In our Streamlit app, we use a simple Neural Network (NN) to classify the images contained in the MNIST dataset. Here is a detailed explanation of each layer in our NN model and its role in the classification process:

### Model Architecture:
1. **Flatten Layer**:
    - **Shape**: (784,)
    - **Function**: This layer flattens the input image (28x28 pixels) into a 1D array of 784 elements. This step is necessary to convert the 2D image data into a format that can be fed into the fully connected (dense) layers.

2. **Dense Layer 1**:
    - **Shape**: (128,)
    - **Units**: 128
    - **Activation**: ReLU
    - **Function**: This fully connected (dense) layer consists of 128 neurons. The ReLU activation function introduces non-linearity, enabling the network to learn complex patterns. This layer processes the flattened input and learns to identify features that are useful for classification.

3. **Dense Layer 2 (Output Layer)**:
    - **Shape**: (10,)
    - **Units**: 10
    - **Activation**: Linear (followed by a softmax operation)
    - **Function**: This output layer consists of 10 neurons, each corresponding to one of the digit classes (0-9). The linear activation function is used here because the subsequent loss function (`SparseCategoricalCrossentropy`) expects logits. During inference, a softmax operation is applied to convert these logits into a probability distribution over the 10 classes.

### Compilation:
- **Optimizer**: Adam
    - The Adam optimizer is used to update the network weights iteratively based on the training data. It combines the advantages of two other extensions of stochastic gradient descent: AdaGrad and RMSProp.
- **Loss Function**: Sparse Categorical Crossentropy (from_logits=True)
    - This loss function is used for classification tasks where the target labels are integers. The `from_logits=True` argument indicates that the output values are raw logits, which will be converted to probabilities internally.
- **Metrics**: Accuracy
    - The accuracy metric is used to evaluate the performance of the model by calculating the proportion of correctly classified images.
"""
)


st.write("---")
st.write("Model Training and Validation")
st.image(
    "/Users/lukewilsen/Desktop/IEX/IEX_Training/Neural_Networks/screenshots/Tensor_Flow_sc.png",
    caption="Train and Validation Accuracy through Epoch Progression",
)

st.write("Graphs of model accuracy through epoch progression")
st.image(
    "/Users/lukewilsen/Desktop/IEX/IEX_Training/Neural_Networks/screenshots/TF_accuracy.png",
    caption="Model Performance",
)
st.image(
    "/Users/lukewilsen/Desktop/IEX/IEX_Training/Neural_Networks/screenshots/TF_mod_loss.png",
    caption="Model Loss",
)

st.write("""
### Summary:
The NN model processes the MNIST images through a series of dense layers. It first flattens the input images, then uses a hidden dense layer to learn patterns, and finally outputs logits for each of the 10 classes. The model learns to recognize patterns and structures in the images, enabling it to accurately classify handwritten digits.
""")