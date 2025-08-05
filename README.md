# 🧠 Fashion MNIST Classification using Deep Neural Networks (Keras)
A complete deep learning project to classify fashion items using the Fashion MNIST dataset.  
This portfolio project demonstrates the end-to-end application of model design, training, evaluation, and testing using **TensorFlow/Keras**.

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

### ✅ Hyperparameter Tuning
- Custom tuning of:
  - **Number of layers and neurons**
  - **Activation functions**
  - **Optimizer learning rate**

### ✅ Regularization Techniques
- **Dropout layers** added after major dense layers to reduce overfitting by randomly deactivating neurons during training.

### ✅ Optimizer Tuning
- Used **Adam** optimizer with a custom learning rate (`0.0005`) for more stable and smoother convergence.

### ✅ Early Stopping
- Implemented **EarlyStopping callback** to halt training once the model stopped improving on validation loss, restoring the best weights automatically.

✅ These methods collectively resulted in a robust model with strong generalization and reduced overfitting.

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
**Epochs**: 30 (with early stopping)

## 📉 Training Results
| Metric              | Value     |
|---------------------|-----------|
| Final Training Acc  | ~91.0%    |
| Final Validation Acc| ~89.3%    |
| Final Test Accuracy | ~89.2%    |
| Params              | 567,434   |
✅ Smooth convergence  
✅ No overfitting due to regularization + early stopping  
✅ Training stopped after 21 epochs with best weights restored

## 📈 Learning Curves
Both training and validation **accuracy/loss** were plotted using `matplotlib`, clearly showing stable learning behavior.

## 🧪 Evaluation on Test Data
The model was evaluated using `.evaluate(X_test, y_test)`  
Sample predictions were visualized on real test images.

Example output:
| Image       | True Label | Predicted |
|-------------|------------|-----------|
| 👟 Shoe      | Ankle boot | Ankle boot ✅ |
| 👕 Sweatshirt | Pullover   | Pullover ✅ |
| 👖 Trousers   | Trouser    | Trouser ✅ |
The model correctly predicted all 3 examples.


## 🧠 Key Features Implemented
- [x] Custom NN architecture with layer tuning
- [x] Dropout layers to reduce overfitting
- [x] Adam optimizer with custom learning rate
- [x] EarlyStopping to halt training intelligently
- [x] Prediction and label mapping
- [x] Grid image visualization using matplotlib
- [x] Reproducibility with random seeds
