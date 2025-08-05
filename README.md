# 🧠 Fashion MNIST Classification using Deep Neural Networks (Keras)
A complete deep learning project to classify fashion items using the Fashion MNIST dataset.  
This portfolio project demonstrates the end-to-end application of model design, training, evaluation, and testing using **TensorFlow/Keras**.

Use this link to access the notebook in Google Colab:  
[Open Colab Notebook](https://github.com/Aishwaryachen11/Fashion_MNIST_Classifier/blob/main/Fashion_MNIST_Portfolio_Project.ipynb)

## 📌 Project Overview
- **Objective**: Build a neural network to classify 28x28 grayscale images into 10 fashion categories.
- **Dataset**: Fashion MNIST (70,000 images total)
- **Tools**: Python, TensorFlow, Keras, NumPy, Matplotlib
- **Techniques**:
  - Neural network design from scratch
  - Dropout regularization
  - Early stopping for overfitting control
  - Learning rate tuning
  - Visualization of learning curves and predictions

## 🔧 Key Techniques Applied
This project goes beyond a basic model by implementing **deep learning best practices** to improve performance and generalization:

### Hyperparameter Tuning
- Manually tuned the number of:
  - Dense layers and neurons (512 → 256 → 128)
  - Activation functions (ReLU for hidden, softmax for output)
  - Learning rate (`0.0005`) for **Adam** optimizer

### Regularization Techniques
- **Dropout layers (30%)** used after major dense layers to mitigate overfitting by randomly deactivating neurons.

### Optimizer Tuning
- Used **Adam** optimizer with a custom learning rate (`0.0005`) for more stable and smoother convergence.

### Early Stopping
- Implemented **EarlyStopping callback** to halt training(patience=3) once the model stopped improving on validation loss, restoring the best weights automatically.
These methods collectively resulted in a robust model with strong generalization and reduced overfitting.

## 🗂️ Dataset Information
The dataset is loaded directly from `keras.datasets.fashion_mnist`.  
It contains:
- 60,000 training images
- 10,000 test images  
Each image is 28x28 pixels, representing one of 10 fashion categories:
We normalize the pixel values to the [0, 1] range by dividing by 255.  
The training set is further split into a **training** and **validation set** (90/10).

## 🏗️ Model Architecture
A custom **feedforward neural network (fully connected)** was designed with:
| Layer        | Units | Activation | Notes                      |
|--------------|-------|------------|----------------------------|
| Flatten      | 784   | —          | Converts 2D to 1D          |
| Dense        | 512   | ReLU       | + Dropout(0.3)             |
| Dense        | 256   | ReLU       | + Dropout(0.3)             |
| Dense        | 128   | ReLU       |                            |
| Output       | 10    | Softmax    | Multiclass classification  |

**Optimizer**: Adam with learning rate = 0.0005  
**Loss Function**: Sparse Categorical Crossentropy  
**Epochs**: 30 (early stopping kicked in at epoch 21)

### 🔍 Model Diagram

<img src="https://github.com/Aishwaryachen11/Fashion_MNIST_Classifier/blob/main/model.png" alt="Model Architecture" width="600"/>

## 📉 Training Results
| Metric              | Value     |
|---------------------|-----------|
| Final Training Acc  | ~91.0%    |
| Final Validation Acc| ~89.3%    |
| Final Test Accuracy | ~89.2%    |
| Params              | 567,434   |
| Epochs Trained      | 21 / 30   |
Training stopped early based on validation performance — indicating **strong generalization** and **low overfitting**.

## 📈 Learning Curves
Both training and validation **accuracy/loss** were plotted using `matplotlib`, clearly showing stable learning behavior.

### Accuracy Over Epochs

<img src="https://github.com/Aishwaryachen11/Fashion_MNIST_Classifier/blob/main/Accuracy_Loss_plot.png" alt="Accuracy over Epochs" width="600"/>

### Loss Over Epochs

<img src="https://github.com/Aishwaryachen11/Fashion_MNIST_Classifier/blob/main/Accuracy_loss_plot2.png" alt="Loss over Epochs" width="600"/>

> Both plots show healthy convergence and no overfitting due to dropout and early stopping.

## 🧪 Evaluation on Test Data
The model was evaluated using `.evaluate(X_test, y_test)`  
Sample predictions were visualized on real test images.

Example output:
| Image       | True Label | Predicted |
|-------------|------------|-----------|
| 👟 Shoe      | Ankle boot | Ankle boot |
| 👕 Sweatshirt | Pullover   | Pullover |
| 👖 Trousers   | Trouser    | Trouser |
The model correctly predicted all 3 examples.


## 🧠 Key Features Implemented
- [x] Custom NN architecture with layer tuning
- [x] Dropout layers to reduce overfitting
- [x] Adam optimizer with custom learning rate
- [x] EarlyStopping to halt training intelligently
- [x] Prediction and label mapping
- [x] Grid image visualization using matplotlib
- [x] Reproducibility with random seeds
