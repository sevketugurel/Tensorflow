# **Bicycle Price Prediction**

This repository contains a simple machine learning model to predict bicycle prices based on specific features. The model is implemented using TensorFlow and utilizes a neural network with the Keras API.

## **Getting Started**

### **Prerequisites**

Make sure you have the following dependencies installed:

- pandas
- seaborn
- matplotlib
- scikit-learn
- TensorFlow

You can install them using the following command:

```bash
bashCopy code
pip install pandas seaborn matplotlib scikit-learn tensorflow

```

### **Dataset**

The dataset used for training and testing the model is loaded from the "bisiklet_fiyatlari.xlsx" Excel file. It includes features like "BisikletOzellik1" and "BisikletOzellik2" along with the target variable "Fiyat" (price).

### **Data Preprocessing**

The dataset is split into training and testing sets using **`train_test_split`** from scikit-learn. Additionally, feature scaling is performed using **`MinMaxScaler`** to ensure numerical stability.

```python
pythonCopy code
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ... (code snippets for data splitting and scaling)

```

### **Model Training**

The neural network model is implemented using TensorFlow's Keras API. It consists of three hidden layers with ReLU activation functions and is trained using the mean squared error (MSE) loss and RMSprop optimizer.

```python
pythonCopy code
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ... (code snippet for model definition and training)

```

### **Model Evaluation**

The model's performance is evaluated using mean squared error (MSE) on both the training and testing sets.

```python
pythonCopy code
from sklearn.metrics import mean_squared_error

# ... (code snippet for model evaluation)

```

### **Model Prediction**

The trained model is used to make predictions on a set of new bicycle features. The results are saved to the "bisiklet_modeli.h5" file for future use.

```python
pythonCopy code
# ... (code snippet for model prediction and saving)

```

## **Results**

The model's predictions are compared with the actual prices, and the performance is visualized using a scatter plot.

```python
pythonCopy code
import seaborn as sbn
import matplotlib.pyplot as plt

# ... (code snippet for visualization)

```

## **Usage**

To run the code, execute the provided Python script. You can also explore and modify the notebook for a more interactive experience.

```bash
bashCopy code
python bicycle_price_prediction.py

```

Feel free to adjust hyperparameters, explore different models, or use alternative datasets to improve the model's performance.
