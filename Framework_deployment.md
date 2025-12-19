# Complete Guide: ML Fundamentals, Frameworks & Deployment

## Table of Contents
1. [Linearity vs Non-linearity](#linearity-vs-non-linearity)
2. [Understanding Non-linearity in Depth](#understanding-non-linearity)
3. [ML Framework Comparison](#ml-framework-comparison)
4. [Flask for ML Deployment](#flask-for-ml-deployment)
5. [End-to-End ML Pipeline](#end-to-end-ml-pipeline)
6. [Advanced Topics](#advanced-topics)
7. [Interview Questions](#interview-questions)

---

## 1. Linearity vs Non-linearity

### What is a Linear Model?

A **linear model** produces output as a weighted sum of input features plus a bias term:

```
y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

**Key Characteristics:**
- Output is a linear combination of inputs
- Decision boundary is a straight line (2D), plane (3D), or hyperplane (n-D)
- Parameters (weights) have a constant effect on output
- Relationship between features and output is additive

**Examples of Linear Models:**
- Linear Regression: `y = 3x₁ + 2x₂ + 5`
- Logistic Regression (linear in log-odds): `logit(p) = w₁x₁ + w₂x₂ + b`
- Support Vector Machines with linear kernel
- Perceptron (single-layer neural network)

### What is a Non-linear Model?

A **non-linear model** cannot be represented as a simple weighted sum. The relationship involves curves, interactions, or transformations.

**Examples of Non-linearity:**

1. **Polynomial features:**
   ```
   y = w₁x₁ + w₂x₁² + w₃x₁³ + b
   ```

2. **Feature interactions:**
   ```
   y = w₁x₁ + w₂x₂ + w₃(x₁ × x₂) + b
   ```

3. **Exponential/logarithmic:**
   ```
   y = w₁exp(x₁) + w₂log(x₂) + b
   ```

4. **Neural networks with activation functions:**
   ```
   y = σ(w₁x₁ + w₂x₂ + b)  where σ is ReLU, sigmoid, etc.
   ```

### Common Misconceptions

**❌ Misconception:** "More features = non-linear model"
```python
# Still LINEAR even with 1000 features
y = w₁x₁ + w₂x₂ + ... + w₁₀₀₀x₁₀₀₀ + b
```

**✓ Reality:** Linearity depends on the **form of the function**, not the number of features.

**❌ Misconception:** "All neural networks are non-linear"
```python
# Linear neural network (no activation functions)
output = W₂(W₁x + b₁) + b₂  # Can be collapsed to single linear transformation
```

**✓ Reality:** Neural networks need **non-linear activation functions** between layers to be truly non-linear.

### When to Use Linear vs Non-linear Models

**Use Linear Models when:**
- Data has approximately linear relationships
- You need interpretability (coefficients show feature importance)
- Dataset is small or has many features (avoid overfitting)
- Computational efficiency is critical
- You need a baseline model

**Use Non-linear Models when:**
- Data shows curved or complex patterns
- Decision boundaries are not straight lines/planes
- Feature interactions are important
- You have sufficient data to avoid overfitting
- Accuracy is more important than interpretability

---

## 2. Understanding Non-linearity in Depth

### Sources of Non-linearity

#### A. Feature Engineering (Polynomial Features)

```python
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# Original features
X = np.array([[2, 3]])

# Create polynomial features (degree=2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# X_poly = [2, 3, 4, 6, 9]
#           ↑  ↑  ↑  ↑  ↑
#          x₁ x₂ x₁² x₁x₂ x₂²
```

**When to use:**
- You suspect quadratic or cubic relationships
- Need to capture curvature in data
- Working with small datasets where deep learning is overkill

#### B. Feature Interactions

```python
# Original model (linear)
y = 2*age + 3*income

# With interaction (non-linear)
y = 2*age + 3*income + 0.5*(age × income)
# Effect of income now DEPENDS on age!
```

**Real-world example:**
- Marketing: Effect of ad spend might depend on seasonality
- Healthcare: Drug effectiveness might depend on patient age
- Finance: Risk might depend on both debt AND income together

#### C. Non-linear Activation Functions

```python
import torch
import torch.nn as nn

class NonLinearNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 64)
        self.relu = nn.ReLU()  # Non-linear activation
        self.layer2 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)  # Introduces non-linearity
        x = self.layer2(x)
        return x
```

**Common Activation Functions:**

| Activation | Formula | Range | Use Case |
|------------|---------|-------|----------|
| ReLU | max(0, x) | [0, ∞) | Hidden layers (most common) |
| Sigmoid | 1/(1+e⁻ˣ) | (0, 1) | Binary classification output |
| Tanh | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | (-1, 1) | Hidden layers (centered data) |
| Softmax | eˣⁱ/Σeˣʲ | (0, 1), sum=1 | Multi-class output |
| Leaky ReLU | max(0.01x, x) | (-∞, ∞) | Prevents dying ReLU |

#### D. Kernel Trick (SVM)

```python
from sklearn.svm import SVC

# Linear kernel (linear decision boundary)
svm_linear = SVC(kernel='linear')

# RBF kernel (non-linear, can create circular boundaries)
svm_rbf = SVC(kernel='rbf', gamma='scale')

# Polynomial kernel (degree controls non-linearity)
svm_poly = SVC(kernel='poly', degree=3)
```

The kernel trick allows linear models to learn non-linear boundaries **without explicitly computing** transformed features.

### Visualizing Linearity vs Non-linearity

**Linear Decision Boundary:**
```
Class A: ●●●●●●
         ●●●●●●  |  ○○○○○○
         ●●●●●●  |  ○○○○○○
              (straight line)
Class B:           ○○○○○○
```

**Non-linear Decision Boundary:**
```
    ○○○○○○○○○
  ○○  ●●●●●  ○○
  ○  ●●●●●●●  ○
  ○  ●●●●●●●  ○
  ○○  ●●●●●  ○○
    ○○○○○○○○○
    (circular/curved)
```

---

## 3. ML Framework Comparison

### Overview Table

| Feature | scikit-learn | TensorFlow | PyTorch |
|---------|--------------|------------|---------|
| **Primary Use** | Classical ML | Production DL | Research DL |
| **Learning Curve** | Easy | Moderate-Hard | Moderate |
| **Data Type** | Tabular | Images, Text, Audio | Images, Text, Audio |
| **Deployment** | Simple (pickle) | TF Serving, TF Lite | TorchServe, ONNX |
| **Execution** | Eager | Graph (+ Eager) | Eager |
| **Best For** | Quick experiments | Mobile/Edge | Prototyping |
| **Company** | Independent | Google | Meta |

### Scikit-learn: Classical ML

**Philosophy:** Simple, consistent API for traditional machine learning.

#### Typical Workflow

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. Load data
X, y = load_data()  # Features and labels

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Preprocess
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# 5. Predict
y_pred = model.predict(X_test_scaled)

# 6. Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

#### Available Models

**Supervised Learning:**
- Linear Models: `LinearRegression`, `LogisticRegression`, `Ridge`, `Lasso`
- Tree-based: `DecisionTreeClassifier`, `RandomForestRegressor`, `GradientBoostingClassifier`
- Support Vector Machines: `SVC`, `SVR`
- Naive Bayes: `GaussianNB`, `MultinomialNB`
- Nearest Neighbors: `KNeighborsClassifier`

**Unsupervised Learning:**
- Clustering: `KMeans`, `DBSCAN`, `AgglomerativeClustering`
- Dimensionality Reduction: `PCA`, `TSNE`, `UMAP` (via umap-learn)
- Anomaly Detection: `IsolationForest`, `OneClassSVM`

**Model Selection:**
- Cross-validation: `cross_val_score`, `GridSearchCV`, `RandomizedSearchCV`
- Metrics: `accuracy_score`, `precision_recall_fscore_support`, `confusion_matrix`

#### When to Use scikit-learn

✅ **Use scikit-learn for:**
- Tabular/structured data (CSV, database exports)
- Small to medium datasets (< 1M rows typically)
- Quick baselines and experiments
- Feature engineering and preprocessing
- Classical algorithms (trees, SVM, k-means)
- When you need interpretable models

❌ **Don't use scikit-learn for:**
- Deep learning (CNNs, RNNs, Transformers)
- Very large datasets (billions of parameters)
- Images, audio, video (use TF/PyTorch)
- Online/streaming learning (limited support)

### TensorFlow: Production Deep Learning

**Philosophy:** Build once, deploy anywhere with production-grade tools.

#### High-level API (Keras)

```python
import tensorflow as tf
from tensorflow import keras

# 1. Build model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# 2. Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 3. Train
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    callbacks=[keras.callbacks.EarlyStopping(patience=3)]
)

# 4. Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)

# 5. Predict
predictions = model.predict(X_new)

# 6. Save
model.save('my_model.h5')
# or
model.save('my_model_folder/')  # SavedModel format
```

#### Low-level API (tf.function)

```python
import tensorflow as tf

# Custom training loop
@tf.function  # Graph compilation for speed
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss

# Training loop
for epoch in range(num_epochs):
    for x_batch, y_batch in dataset:
        loss = train_step(x_batch, y_batch)
```

#### TensorFlow Ecosystem

**TensorFlow Extended (TFX):** End-to-end ML pipelines
**TensorFlow Lite:** Mobile and embedded devices
**TensorFlow.js:** Browser-based ML
**TensorFlow Serving:** Production model serving
**TensorFlow Hub:** Pre-trained models

#### When to Use TensorFlow

✅ **Use TensorFlow for:**
- Production deployments (mobile, edge, cloud)
- Large-scale distributed training
- Multi-language support (Python, Java, C++, JavaScript)
- When you need TensorBoard for visualization
- Keras high-level API for standard architectures
- When deploying to Google Cloud (Vertex AI)

❌ **Consider alternatives for:**
- Research and experimentation (PyTorch is often easier)
- When you need maximum flexibility
- Quick prototyping (can feel heavy compared to PyTorch)

### PyTorch: Research and Prototyping

**Philosophy:** Pythonic, intuitive, and flexible deep learning.

#### Typical Workflow

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1. Define model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 2. Initialize
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size=784, hidden_size=128, num_classes=10).to(device)

# 3. Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Prepare data
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 5. Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Step [{i+1}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}')

# 6. Save model
torch.save(model.state_dict(), 'model.pth')

# 7. Load model
model = NeuralNet(784, 128, 10)
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

#### PyTorch Advantages

**Dynamic Computation Graphs:**
```python
# PyTorch: graph built on-the-fly, easy to debug
for epoch in range(10):
    output = model(x)  # Can change model structure here!
    loss = criterion(output, y)
    loss.backward()
    
    # Easy to inspect gradients
    print(model.fc1.weight.grad)  # Just normal Python debugging!
```

**Pythonic Feel:**
```python
# PyTorch tensors behave like NumPy arrays
x = torch.randn(3, 4)
print(x[0, :])  # Slicing works as expected
print(x.mean())  # Familiar operations
x.numpy()  # Easy conversion to NumPy
```

#### PyTorch Ecosystem

**torchvision:** Computer vision models and datasets
**torchaudio:** Audio processing
**torchtext:** NLP utilities (deprecated, use HuggingFace)
**PyTorch Lightning:** High-level wrapper for cleaner code
**HuggingFace Transformers:** Built on PyTorch (primarily)

#### When to Use PyTorch

✅ **Use PyTorch for:**
- Research and experimentation
- Custom architectures and novel ideas
- When you want to understand what's happening
- Academic projects
- Fast prototyping
- When using HuggingFace Transformers
- Computer vision research

❌ **Consider alternatives for:**
- When you need native mobile deployment (TF Lite is more mature)
- When production deployment is the priority
- If your team is already using TensorFlow

### Framework Decision Tree

```
START
  |
  ├─ Tabular/Structured Data?
  |    ├─ YES → scikit-learn
  |    └─ NO → Continue
  |
  ├─ Deep Learning Needed?
  |    ├─ NO → scikit-learn
  |    └─ YES → Continue
  |
  ├─ Production Deployment Critical?
  |    ├─ YES → TensorFlow
  |    └─ NO → Continue
  |
  ├─ Research/Experimentation?
  |    ├─ YES → PyTorch
  |    └─ MAYBE → Continue
  |
  ├─ Need Mobile/Edge Deployment?
  |    ├─ YES → TensorFlow
  |    └─ NO → PyTorch (for flexibility)
  |
  └─ Using HuggingFace?
       └─ YES → PyTorch
```

---

## 4. Flask for ML Deployment

### What is Flask?

Flask is a **lightweight web framework** for Python that allows you to build web applications and APIs.

**Role in ML:** Flask serves as the **bridge** between your trained ML model and the outside world (users, apps, services).

### Flask Basics

#### Minimal Flask App

```python
from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, World!"

if __name__ == '__main__':
    app.run(debug=True)
```

#### Understanding Routes

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

# GET request (read data)
@app.route('/items', methods=['GET'])
def get_items():
    items = ["apple", "banana", "orange"]
    return jsonify(items)

# POST request (send data)
@app.route('/items', methods=['POST'])
def create_item():
    data = request.get_json()
    new_item = data.get('name')
    return jsonify({"message": f"Created {new_item}"}), 201

# Dynamic route (with parameters)
@app.route('/items/<int:item_id>', methods=['GET'])
def get_item(item_id):
    return jsonify({"id": item_id, "name": "apple"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

### Flask for ML: Complete Example

#### Step 1: Train and Save Model

```python
# train_model.py
import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Save model
with open('iris_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print(f"Model accuracy: {model.score(X_test, y_test):.2f}")
```

#### Step 2: Create Flask API

```python
# app.py
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model on startup
with open('iris_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return "Iris Classification API"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Extract features
        features = np.array([[
            data['sepal_length'],
            data['sepal_width'],
            data['petal_length'],
            data['petal_width']
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0].tolist()
        
        # Map prediction to class name
        class_names = ['setosa', 'versicolor', 'virginica']
        
        return jsonify({
            'prediction': class_names[prediction],
            'probabilities': {
                'setosa': probability[0],
                'versicolor': probability[1],
                'virginica': probability[2]
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

#### Step 3: Test the API

**Using curl:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```

**Using Python requests:**
```python
import requests

url = 'http://localhost:5000/predict'
data = {
    'sepal_length': 5.1,
    'sepal_width': 3.5,
    'petal_length': 1.4,
    'petal_width': 0.2
}

response = requests.post(url, json=data)
print(response.json())
# Output: {'prediction': 'setosa', 'probabilities': {...}}
```

### Advanced Flask for ML

#### Input Validation

```python
from flask import Flask, request, jsonify

def validate_input(data):
    required_fields = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    
    for field in required_fields:
        if field not in data:
            return False, f"Missing field: {field}"
        
        try:
            value = float(data[field])
            if value < 0 or value > 10:
                return False, f"{field} must be between 0 and 10"
        except ValueError:
            return False, f"{field} must be a number"
    
    return True, None

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    is_valid, error_message = validate_input(data)
    if not is_valid:
        return jsonify({'error': error_message}), 400
    
    # Continue with prediction...
```

#### Batch Predictions

```python
@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    data = request.get_json()
    samples = data.get('samples', [])
    
    if not samples:
        return jsonify({'error': 'No samples provided'}), 400
    
    # Convert to numpy array
    features = np.array([[
        s['sepal_length'],
        s['sepal_width'],
        s['petal_length'],
        s['petal_width']
    ] for s in samples])
    
    # Make predictions
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    
    class_names = ['setosa', 'versicolor', 'virginica']
    results = []
    
    for i, pred in enumerate(predictions):
        results.append({
            'prediction': class_names[pred],
            'probabilities': {
                'setosa': probabilities[i][0],
                'versicolor': probabilities[i][1],
                'virginica': probabilities[i][2]
            }
        })
    
    return jsonify({'results': results})
```

#### Model Versioning

```python
import os
from datetime import datetime

MODEL_DIR = 'models'
CURRENT_MODEL = 'v1'

def load_model(version='v1'):
    model_path = os.path.join(MODEL_DIR, f'model_{version}.pkl')
    with open(model_path, 'rb') as f:
        return pickle.load(f)

# Global model
model = load_model(CURRENT_MODEL)

@app.route('/model/version', methods=['GET'])
def get_version():
    return jsonify({'version': CURRENT_MODEL})

@app.route('/model/reload', methods=['POST'])
def reload_model():
    global model, CURRENT_MODEL
    data = request.get_json()
    version = data.get('version', 'v1')
    
    try:
        model = load_model(version)
        CURRENT_MODEL = version
        return jsonify({
            'message': f'Model reloaded to version {version}',
            'timestamp': datetime.now().isoformat()
        })
    except FileNotFoundError:
        return jsonify({'error': f'Model version {version} not found'}), 404
```

### Serving Deep Learning Models

#### PyTorch Model Serving

```python
# app.py
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np

app = Flask(__name__)

# Define model architecture
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 3)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet().to(device)
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Prepare input
    features = torch.tensor([[
        data['sepal_length'],
        data['sepal_width'],
        data['petal_length'],
        data['petal_width']
    ]], dtype=torch.float32).to(device)
    
    # Predict (no gradient computation needed)
    with torch.no_grad():
        output = model(features)
        probabilities = torch.softmax(output, dim=1)[0]
        prediction = torch.argmax(probabilities).item()
    
    class_names = ['setosa', 'versicolor', 'virginica']
    
    return jsonify({
        'prediction': class_names[prediction],
        'probabilities': {
            'setosa': probabilities[0].item(),
            'versicolor': probabilities[1].item(),
            'virginica': probabilities[2].item()
        }
    })

if __name__ == '__main__':
    app.run(debug=True)
```

#### TensorFlow/Keras Model Serving

```python
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model('iris_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Prepare input
    features = np.array([[
        data['sepal_length'],
        data['sepal_width'],
        data['petal_length'],
        data['petal_width']
    ]])
    
    # Predict
    prediction = model.predict(features)
    predicted_class = np.argmax(prediction[0])
    
    class_names = ['setosa', 'versicolor', 'virginica']
    
    return jsonify({
        'prediction': class_names[predicted_class],
        'probabilities': {
            'setosa': float(prediction[0][0]),
            'versicolor': float(prediction[0][1]),
            'virginica': float(prediction[0][2])
        }
    })

if __name__ == '__main__':
    app.run(debug=True)
```

### Production Considerations

#### 1. CORS (Cross-Origin Resource Sharing)

```python
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow all origins

# Or be more specific
CORS(app, resources={r"/api/*": {"origins": "https://myapp.com"}})
```

#### 2. Error Handling

```python
from flask import Flask, jsonify

app = Flask(__name__)

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({'error': str(e)}), 500
```

#### 3. Logging

```python
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
app.logger.addHandler(handler)

@app.route('/predict', methods=['POST'])
def predict():
    app.logger.info('Prediction request received')
    try:
        # ... prediction code
        app.logger.info('Prediction successful')
        return jsonify(result)
    except Exception as e:
        app.logger.error(f'Prediction failed: {str(e)}')
        return jsonify({'error': str(e)}), 500
```

#### 4. Rate Limiting

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict():
    # Prediction code
    pass
```

#### 5. Authentication

```python
from functools import wraps
from flask import request, jsonify

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != 'your-secret-api-key':
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/predict', methods=['POST'])
@require_api_key
def predict():
    # Prediction code
    pass
```

---

## 5. End-to-End ML Pipeline

### Complete Workflow

```
1. Data Collection & Exploration
        ↓
2. Data Preprocessing & Feature Engineering
        ↓
3. Model Training & Validation
        ↓
4. Model Evaluation & Selection
        ↓
5. Model Serialization (Save)
        ↓
6. API Development (Flask)
        ↓
7. Testing & Documentation
        ↓
8. Deployment (Cloud/Server)
        ↓
9. Monitoring & Maintenance
```

### Step-by-Step Implementation

#### Step 1: Data Collection & Exploration

```python
# data_exploration.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('data.csv')

# Basic exploration
print(df.head())
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Visualize distributions
df.hist(figsize=(12, 8))
plt.tight_layout()
plt.savefig('distributions.png')

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.savefig('correlation.png')
```

#### Step 2: Data Preprocessing

```python
# preprocessing.py
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

def preprocess_data(df):
    # Handle missing values
    df = df.dropna()  # or df.fillna(df.mean())
    
    # Encode categorical variables
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoders

# Save preprocessors
import pickle
scaler, label_encoders = preprocess_data(df)[4:6]
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
```

#### Step 3: Model Training

```python
# train.py
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import pickle

def train_multiple_models(X_train, y_train):
    models = {
        'logistic_regression': LogisticRegression(max_iter=1000),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        # Cross-validation
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        results[name] = {
            'mean_score': scores.mean(),
            'std_score': scores.std()
        }
        
        # Train on full training set
        model.fit(X_train, y_train)
        
        # Save model
        with open(f'{name}_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        print(f"{name}: {scores.mean():.3f} (+/- {scores.std():.3f})")
    
    return results

# Train models
results = train_multiple_models(X_train_scaled, y_train)

# Select best model
best_model_name = max(results, key=lambda k: results[k]['mean_score'])
print(f"\nBest model: {best_model_name}")
```

#### Step 4: Model Evaluation

```python
# evaluate.py
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import precision_recall_curve, roc_curve
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    
    # ROC curve (for binary classification)
    if len(np.unique(y_test)) == 2:
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        auc = roc_auc_score(y_test, y_proba[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig('roc_curve.png')
    
    return y_pred, y_proba

# Load best model and evaluate
with open('random_forest_model.pkl', 'rb') as f:
    best_model = pickle.load(f)

y_pred, y_proba = evaluate_model(best_model, X_test_scaled, y_test)
```

#### Step 5: Production Flask API

```python
# production_app.py
from flask import Flask, request, jsonify
import pickle
import numpy as np
import logging
from datetime import datetime

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and preprocessors
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Model metadata
MODEL_VERSION = "1.0.0"
FEATURE_NAMES = ['feature1', 'feature2', 'feature3', 'feature4']

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'service': 'ML Prediction API',
        'version': MODEL_VERSION,
        'endpoints': {
            '/predict': 'POST - Make a prediction',
            '/predict_batch': 'POST - Batch predictions',
            '/health': 'GET - Health check',
            '/model_info': 'GET - Model information'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None
    })

@app.route('/model_info', methods=['GET'])
def model_info():
    return jsonify({
        'version': MODEL_VERSION,
        'algorithm': type(model).__name__,
        'features': FEATURE_NAMES,
        'n_features': len(FEATURE_NAMES)
    })

def validate_features(data):
    """Validate input features"""
    for feature in FEATURE_NAMES:
        if feature not in data:
            return False, f"Missing feature: {feature}"
    return True, None

def preprocess_input(data):
    """Preprocess input data for prediction"""
    features = np.array([[data[f] for f in FEATURE_NAMES]])
    features_scaled = scaler.transform(features)
    return features_scaled

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info("Received prediction request")
        
        # Get and validate data
        data = request.get_json()
        is_valid, error_msg = validate_features(data)
        if not is_valid:
            logger.warning(f"Invalid input: {error_msg}")
            return jsonify({'error': error_msg}), 400
        
        # Preprocess
        features = preprocess_input(data)
        
        # Predict
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        # Prepare response
        response = {
            'prediction': int(prediction),
            'probability': float(probability[int(prediction)]),
            'all_probabilities': {
                str(i): float(prob) for i, prob in enumerate(probability)
            },
            'timestamp': datetime.now().isoformat(),
            'model_version': MODEL_VERSION
        }
        
        logger.info(f"Prediction successful: {prediction}")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        logger.info("Received batch prediction request")
        
        # Get data
        data = request.get_json()
        samples = data.get('samples', [])
        
        if not samples:
            return jsonify({'error': 'No samples provided'}), 400
        
        # Validate all samples
        for i, sample in enumerate(samples):
            is_valid, error_msg = validate_features(sample)
            if not is_valid:
                return jsonify({
                    'error': f"Sample {i}: {error_msg}"
                }), 400
        
        # Preprocess all samples
        features_list = [preprocess_input(sample) for sample in samples]
        features_batch = np.vstack(features_list)
        
        # Batch predict
        predictions = model.predict(features_batch)
        probabilities = model.predict_proba(features_batch)
        
        # Prepare response
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                'prediction': int(pred),
                'probability': float(prob[int(pred)]),
                'all_probabilities': {
                    str(j): float(p) for j, p in enumerate(prob)
                }
            })
        
        response = {
            'predictions': results,
            'count': len(results),
            'timestamp': datetime.now().isoformat(),
            'model_version': MODEL_VERSION
        }
        
        logger.info(f"Batch prediction successful: {len(results)} samples")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
```

#### Step 6: Frontend Integration (React Example)

```javascript
// api.js
const API_BASE_URL = 'http://localhost:5000';

export async function makePrediction(features) {
  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(features),
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Prediction failed');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Prediction error:', error);
    throw error;
  }
}

// PredictionForm.jsx
import React, { useState } from 'react';

function PredictionForm() {
  const [features, setFeatures] = useState({
    feature1: '',
    feature2: '',
    feature3: '',
    feature4: '',
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    
    try {
      const prediction = await makePrediction(features);
      setResult(prediction);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div>
      <form onSubmit={handleSubmit}>
        {Object.keys(features).map((key) => (
          <input
            key={key}
            type="number"
            placeholder={key}
            value={features[key]}
            onChange={(e) => setFeatures({
              ...features,
              [key]: e.target.value
            })}
          />
        ))}
        <button type="submit" disabled={loading}>
          {loading ? 'Predicting...' : 'Predict'}
        </button>
      </form>
      
      {error && <div className="error">{error}</div>}
      
      {result && (
        <div className="result">
          <h3>Prediction: {result.prediction}</h3>
          <p>Confidence: {(result.probability * 100).toFixed(2)}%</p>
        </div>
      )}
    </div>
  );
}
```

---

## 6. Advanced Topics

### Model Optimization Techniques

#### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Grid Search (exhaustive)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.3f}")

# Random Search (faster for large parameter spaces)
from scipy.stats import randint, uniform

param_distributions = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(10, 50),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)
```

#### Feature Selection

```python
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier

# Univariate feature selection
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X_train, y_train)
selected_features = selector.get_support(indices=True)
print(f"Selected features: {selected_features}")

# Recursive Feature Elimination
model = RandomForestClassifier(n_estimators=100)
rfe = RFE(estimator=model, n_features_to_select=10)
rfe.fit(X_train, y_train)
print(f"Feature ranking: {rfe.ranking_}")

# Feature importance from tree-based models
model.fit(X_train, y_train)
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices])
plt.show()
```

#### Model Ensemble Techniques

```python
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Voting Classifier (Hard voting)
clf1 = LogisticRegression(max_iter=1000)
clf2 = RandomForestClassifier(n_estimators=100)
clf3 = SVC(probability=True)

voting_clf = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('svc', clf3)],
    voting='soft'  # 'soft' uses probabilities, 'hard' uses predictions
)
voting_clf.fit(X_train, y_train)

# Stacking Classifier
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svc', SVC(probability=True, random_state=42))
]
stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression()
)
stacking_clf.fit(X_train, y_train)
```

### Model Monitoring & Drift Detection

```python
# drift_detection.py
from scipy import stats
import numpy as np

def detect_data_drift(reference_data, current_data, threshold=0.05):
    """
    Detect data drift using Kolmogorov-Smirnov test
    """
    drift_detected = {}
    
    for feature_idx in range(reference_data.shape[1]):
        ref_feature = reference_data[:, feature_idx]
        curr_feature = current_data[:, feature_idx]
        
        # KS test
        statistic, p_value = stats.ks_2samp(ref_feature, curr_feature)
        
        drift_detected[f'feature_{feature_idx}'] = {
            'drift': p_value < threshold,
            'p_value': p_value,
            'statistic': statistic
        }
    
    return drift_detected

def detect_model_drift(model, X_train, X_current, y_current, threshold=0.1):
    """
    Detect model performance drift
    """
    train_accuracy = model.score(X_train, y_train)
    current_accuracy = model.score(X_current, y_current)
    
    accuracy_drop = train_accuracy - current_accuracy
    
    return {
        'train_accuracy': train_accuracy,
        'current_accuracy': current_accuracy,
        'accuracy_drop': accuracy_drop,
        'drift_detected': accuracy_drop > threshold
    }
```

### A/B Testing for Models

```python
# ab_testing.py
import random
from collections import defaultdict

class ABTestingRouter:
    def __init__(self, model_a, model_b, split=0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.split = split
        self.results = defaultdict(list)
    
    def predict(self, X, user_id=None):
        # Assign to model A or B
        if random.random() < self.split:
            model_name = 'A'
            prediction = self.model_a.predict(X)
        else:
            model_name = 'B'
            prediction = self.model_b.predict(X)
        
        # Log for analysis
        self.results[model_name].append({
            'user_id': user_id,
            'prediction': prediction
        })
        
        return prediction, model_name
    
    def analyze_results(self):
        """Compare model performance"""
        return {
            'model_a_count': len(self.results['A']),
            'model_b_count': len(self.results['B']),
            # Add more metrics (conversion rate, accuracy, etc.)
        }
```

---

## 7. Interview Questions

### Beginner Level

**Q1: What's the difference between a linear and non-linear model?**

**A:** A linear model produces output as a weighted sum of input features (y = w₁x₁ + w₂x₂ + b), creating straight decision boundaries in feature space. A non-linear model involves transformations like squares (x²), products (x₁×x₂), or non-linear activation functions, allowing it to learn curved decision boundaries and capture complex patterns that linear models cannot.

---

**Q2: Does having more features make a model non-linear?**

**A:** No. The number of features doesn't determine linearity. A model with 1000 features can still be linear if it's a weighted sum (y = w₁x₁ + w₂x₂ + ... + w₁₀₀₀x₁₀₀₀). Linearity depends on the **form of the function**, not feature count. Non-linearity comes from polynomial terms, interactions, or non-linear activation functions.

---

**Q3: When would you use scikit-learn vs PyTorch/TensorFlow?**

**A:** 
- **Use scikit-learn** for: tabular data, classical ML algorithms (random forests, SVM, logistic regression), quick baselines, small-medium datasets, when interpretability matters
- **Use PyTorch/TensorFlow** for: deep learning, image/text/audio data, very large datasets, complex neural networks, when you need GPUs for training

---

**Q4: What is Flask and why is it used in ML?**

**A:** Flask is a lightweight Python web framework that creates REST APIs. In ML, Flask serves as the bridge between trained models and end users. You load your saved model into a Flask app, create endpoints like `/predict`, and users can send data via HTTP requests to get predictions. It's essential for deploying models to production where applications can access them.

---

**Q5: How do you save and load a trained model?**

**A:**
```python
# Scikit-learn (using pickle)
import pickle
pickle.dump(model, open('model.pkl', 'wb'))
loaded_model = pickle.load(open('model.pkl', 'rb'))

# PyTorch
torch.save(model.state_dict(), 'model.pth')
model.load_state_dict(torch.load('model.pth'))

# TensorFlow/Keras
model.save('model.h5')
loaded_model = tf.keras.models.load_model('model.h5')
```

---

### Intermediate Level

**Q6: Explain how non-linear activation functions work in neural networks.**

**A:** Non-linear activation functions (ReLU, sigmoid, tanh) are applied element-wise after each linear transformation in a neural network. Without them, stacking multiple linear layers would collapse into a single linear transformation: W₂(W₁x + b₁) + b₂ = Wx + b (still linear). Non-linear activations break this by introducing curves:
- ReLU(x) = max(0, x) creates piecewise linear functions
- Sigmoid(x) = 1/(1+e⁻ˣ) creates smooth S-curves

This allows networks to approximate any continuous function (universal approximation theorem).

---

**Q7: What's the difference between eager and graph execution in deep learning frameworks?**

**A:**
- **Eager execution** (PyTorch default, TF 2.x): Code runs line-by-line like normal Python. Easy to debug with breakpoints, print statements work immediately. More intuitive but potentially slower.
- **Graph execution** (TF 1.x, @tf.function): Builds a computational graph first, then executes it. Harder to debug but optimizable for production deployment, can run on various backends (CPU, GPU, TPU), and supports model serving infrastructure.

PyTorch uses eager by default (Pythonic), TensorFlow 2.x uses eager but can compile to graphs with @tf.function for performance.

---

**Q8: How do you handle versioning when deploying ML models?**

**A:**
```python
# Approach 1: Directory structure
models/
  v1/model.pkl
  v2/model.pkl
  v3/model.pkl

# Approach 2: Metadata in model file
model_metadata = {
    'version': '2.1.0',
    'trained_date': '2025-01-15',
    'features': ['f1', 'f2', 'f3'],
    'accuracy': 0.95
}

# Approach 3: Model registry (MLflow)
import mlflow
mlflow.register_model("runs:/<run_id>/model", "MyModel")

# In Flask, support multiple versions
@app.route('/predict/<version>', methods=['POST'])
def predict(version):
    model = load_model(version)
    return jsonify(model.predict(data))
```

---

**Q9: What preprocessing must be consistent between training and deployment?**

**A:** ALL preprocessing must match exactly:
1. **Scaling/Normalization**: Save the fitted scaler and apply same transformation
2. **Encoding**: Save label encoders for categorical variables
3. **Feature engineering**: Apply same feature creation logic
4. **Missing value handling**: Use same imputation strategy
5. **Feature order**: Maintain exact same column order

```python
# Training
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
pickle.dump(scaler, open('scaler.pkl', 'wb'))

# Deployment
scaler = pickle.load(open('scaler.pkl', 'rb'))
X_new_scaled = scaler.transform(X_new)  # Use transform, not fit_transform!
```

---

**Q10: How would you implement batch predictions in Flask?**

**A:**
```python
@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    data = request.get_json()
    samples = data['samples']  # List of feature dicts
    
    # Convert to array
    features = np.array([[s[f] for f in FEATURE_NAMES] 
                         for s in samples])
    
    # Preprocess once
    features_scaled = scaler.transform(features)
    
    # Batch predict (much faster than loop)
    predictions = model.predict(features_scaled)
    probabilities = model.predict_proba(features_scaled)
    
    return jsonify({
        'predictions': predictions.tolist(),
        'probabilities': probabilities.tolist()
    })
```

---

### Advanced Level

**Q11: Explain the universal approximation theorem and its practical limitations.**

**A:** The universal approximation theorem states that a neural network with a single hidden layer containing enough neurons can approximate any continuous function to arbitrary precision. However, practical limitations include:
1. **Width vs Depth**: While theoretically true, deeper networks learn hierarchical features more efficiently than very wide shallow networks
2. **Training difficulty**: Having enough capacity doesn't guarantee you can train it effectively (vanishing gradients, local minima)
3. **Overfitting**: Sufficient capacity to approximate anything also means capacity to overfit training data
4. **Computational cost**: May need impractically many neurons in practice

Modern deep learning uses deep networks because depth provides better generalization and feature learning despite narrower layers.

---

**Q12: How do you handle model serving at scale with high traffic?**

**A:** Several strategies:
1. **Horizontal scaling**: Deploy multiple Flask instances behind a load balancer (nginx, AWS ALB)
2. **Model caching**: Keep model in memory, use Redis for feature caching
3. **Batch inference**: Accumulate requests and process in batches
4. **Asynchronous processing**: Use Celery + Redis for async predictions
5. **Specialized serving**: TensorFlow Serving, TorchServe (optimized for DL models)
6. **Model compression**: Quantization, pruning to reduce inference time

```python
# Using gunicorn for production (multiple workers)
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Or uWSGI
uwsgi --http :5000 --wsgi-file app.py --callable app --processes 4

# With async (Celery)
from celery import Celery

celery = Celery('tasks', broker='redis://localhost:6379')

@celery.task
def async_predict(features):
    return model.predict(features)

@app.route('/predict_async', methods=['POST'])
def predict_async():
    task = async_predict.delay(features)
    return jsonify({'task_id': task.id})
```

---

**Q13: What's the difference between L1 and L2 regularization, and when do they introduce non-linearity?**

**A:** 
- **L1 (Lasso)**: Adds sum of absolute values of weights (λΣ|wᵢ|). Creates sparse models (many weights = 0). Useful for feature selection.
- **L2 (Ridge)**: Adds sum of squared weights (λΣwᵢ²). Shrinks all weights proportionally, never exactly zero.

**Key Point**: Regularization itself doesn't make the model non-linear. Linear regression with L1/L2 is still linear (y = wx + b). However:
- Regularization affects the learned weights
- The decision boundary remains a hyperplane
- Non-linearity comes from feature transformations or activation functions, NOT from regularization

```python
# Still linear despite regularization
from sklearn.linear_model import Ridge, Lasso
ridge = Ridge(alpha=1.0)  # L2 regularization
lasso = Lasso(alpha=1.0)  # L1 regularization
# Both produce: y = w₁x₁ + w₂x₂ + ... + b (linear!)
```

---

**Q14: How do you detect and handle data drift in production?**

**A:** Data drift occurs when input data distribution changes over time, degrading model performance.

**Detection methods:**
1. **Statistical tests**: Kolmogorov-Smirnov, Chi-squared test
2. **Distribution monitoring**: Compare feature distributions (training vs production)
3. **Model performance monitoring**: Track accuracy, precision, recall over time
4. **Prediction distribution**: Monitor output distribution changes

```python
from scipy.stats import ks_2samp
import numpy as np

def detect_drift(reference_data, current_data, threshold=0.05):
    """Detect drift using KS test"""
    drifted_features = []
    
    for i in range(reference_data.shape[1]):
        statistic, p_value = ks_2samp(
            reference_data[:, i], 
            current_data[:, i]
        )
        
        if p_value < threshold:
            drifted_features.append(i)
    
    return len(drifted_features) > 0, drifted_features

# Handling drift
# 1. Retrain model with recent data
# 2. Update preprocessing parameters
# 3. Alert monitoring system
# 4. Gradual rollback if performance degrades
```

**Production implementation:**
```python
from datetime import datetime, timedelta

class DriftMonitor:
    def __init__(self, reference_data, window_size=1000):
        self.reference_data = reference_data
        self.current_window = []
        self.window_size = window_size
        self.drift_alerts = []
    
    def add_prediction(self, features):
        self.current_window.append(features)
        
        if len(self.current_window) >= self.window_size:
            # Check for drift
            current_data = np.array(self.current_window)
            drift_detected, features = detect_drift(
                self.reference_data, 
                current_data
            )
            
            if drift_detected:
                self.drift_alerts.append({
                    'timestamp': datetime.now(),
                    'drifted_features': features
                })
                # Trigger retraining or alert
            
            # Reset window
            self.current_window = []
```

---

**Q15: Explain the trade-offs between model complexity and interpretability.**

**A:**

| Aspect | Simple Models | Complex Models |
|--------|---------------|----------------|
| **Interpretability** | High (can explain each prediction) | Low (black box) |
| **Performance** | Lower (underfitting risk) | Higher (can capture complex patterns) |
| **Training time** | Fast | Slow (especially deep learning) |
| **Data requirements** | Less data needed | Requires large datasets |
| **Debugging** | Easy to diagnose issues | Difficult to debug |
| **Stakeholder trust** | Easier to gain buy-in | Harder to explain to non-technical users |

**Real-world decision framework:**
```python
def choose_model(context):
    if context['regulated_industry']:  # Healthcare, finance
        return 'interpretable'  # Logistic regression, decision trees
    
    if context['prediction_accuracy_critical']:  # Self-driving cars
        return 'complex'  # Deep learning
    
    if context['need_to_explain_decisions']:  # Loan approval
        return 'interpretable'  # Can use SHAP for some interpretability
    
    if context['large_unstructured_data']:  # Images, text
        return 'complex'  # Neural networks
    
    return 'start_simple_then_increase'  # General best practice
```

**Making complex models interpretable:**
```python
import shap
import lime

# SHAP (SHapley Additive exPlanations)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

# LIME (Local Interpretable Model-agnostic Explanations)
from lime.lime_tabular import LimeTabularExplainer

explainer = LimeTabularExplainer(
    X_train, 
    feature_names=feature_names,
    class_names=class_names
)
explanation = explainer.explain_instance(X_test[0], model.predict_proba)
```

---

**Q16: How do you implement feature engineering for non-linear patterns?**

**A:** Multiple approaches to capture non-linearity:

**1. Polynomial Features:**
```python
from sklearn.preprocessing import PolynomialFeatures

# Generate x², x³, x₁x₂, etc.
poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly.fit_transform(X)

# Example: [x₁, x₂] → [x₁, x₂, x₁², x₁x₂, x₂², x₁³, x₁²x₂, x₁x₂², x₂³]
```

**2. Domain-specific transformations:**
```python
# Time-based features
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Ratio features (often reveal non-linear relationships)
df['debt_to_income'] = df['debt'] / (df['income'] + 1)  # +1 to avoid division by zero

# Log transformations (for skewed distributions)
df['log_income'] = np.log1p(df['income'])

# Binning (discretization)
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 100], labels=['young', 'adult', 'middle', 'senior'])
```

**3. Feature interactions:**
```python
# Manual interactions
df['price_x_quantity'] = df['price'] * df['quantity']
df['feature1_x_feature2'] = df['feature1'] * df['feature2']

# Automatic interaction detection
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=True)
# Only generates x₁x₂, not x₁², x₂²
```

**4. Target encoding (for categorical variables):**
```python
# Mean encoding (use with cross-validation to avoid leakage!)
category_means = train.groupby('category')['target'].mean()
df['category_encoded'] = df['category'].map(category_means)
```

---

**Q17: Compare synchronous vs asynchronous ML inference.**

**A:**

**Synchronous (Request-Response):**
```python
@app.route('/predict', methods=['POST'])
def predict_sync():
    data = request.get_json()
    # Client waits here
    prediction = model.predict(data)  # Could take 500ms
    return jsonify({'prediction': prediction})
```

**Pros:**
- Simple to implement
- Immediate results
- Easy to debug

**Cons:**
- Client blocked during inference
- Doesn't scale for slow models
- Timeouts on long-running predictions

**Asynchronous (Task Queue):**
```python
from celery import Celery
from flask import Flask, request, jsonify

app = Flask(__name__)
celery = Celery('tasks', broker='redis://localhost:6379')

@celery.task
def predict_async(data):
    time.sleep(5)  # Simulate slow model
    return model.predict(data)

@app.route('/predict', methods=['POST'])
def submit_prediction():
    data = request.get_json()
    task = predict_async.delay(data)
    return jsonify({'task_id': task.id}), 202

@app.route('/result/<task_id>', methods=['GET'])
def get_result(task_id):
    task = predict_async.AsyncResult(task_id)
    
    if task.state == 'PENDING':
        return jsonify({'status': 'pending'}), 202
    elif task.state == 'SUCCESS':
        return jsonify({'status': 'complete', 'result': task.result}), 200
    else:
        return jsonify({'status': 'failed', 'error': str(task.info)}), 500
```

**Pros:**
- Non-blocking
- Better for slow models
- Can prioritize tasks
- Scales horizontally

**Cons:**
- More complex architecture
- Requires message queue (Redis, RabbitMQ)
- Client must poll or use webhooks

**When to use:**
- Sync: Fast models (<100ms), real-time requirements, simple use cases
- Async: Slow models (>1s), batch processing, high traffic, complex workflows

---

**Q18: How do you implement A/B testing for ML models?**

**A:** A/B testing compares two model versions to determine which performs better in production.

```python
# ab_test.py
import random
from datetime import datetime
from collections import defaultdict

class ABTestController:
    def __init__(self, model_a, model_b, split_ratio=0.5):
        """
        Args:
            model_a: Control model (current production)
            model_b: Variant model (new candidate)
            split_ratio: Percentage traffic to model B (0.5 = 50/50)
        """
        self.model_a = model_a
        self.model_b = model_b
        self.split_ratio = split_ratio
        
        # Track metrics
        self.metrics = {
            'A': defaultdict(list),
            'B': defaultdict(list)
        }
    
    def predict(self, features, user_id=None):
        """Route request to model A or B"""
        # Consistent assignment for same user
        if user_id:
            assigned_model = 'B' if hash(user_id) % 100 < self.split_ratio * 100 else 'A'
        else:
            assigned_model = 'B' if random.random() < self.split_ratio else 'A'
        
        # Make prediction
        model = self.model_b if assigned_model == 'B' else self.model_a
        prediction = model.predict(features)
        
        # Log assignment
        self.log_prediction(assigned_model, user_id, features, prediction)
        
        return prediction, assigned_model
    
    def log_prediction(self, model_name, user_id, features, prediction):
        """Log prediction for analysis"""
        self.metrics[model_name]['predictions'].append({
            'timestamp': datetime.now(),
            'user_id': user_id,
            'features': features,
            'prediction': prediction
        })
    
    def record_outcome(self, model_name, user_id, actual_outcome):
        """Record actual outcome (e.g., user clicked, converted, etc.)"""
        self.metrics[model_name]['outcomes'].append({
            'timestamp': datetime.now(),
            'user_id': user_id,
            'outcome': actual_outcome
        })
    
    def get_statistics(self):
        """Calculate A/B test statistics"""
        stats = {}
        
        for model_name in ['A', 'B']:
            n_predictions = len(self.metrics[model_name]['predictions'])
            n_positive_outcomes = sum(
                1 for o in self.metrics[model_name]['outcomes'] 
                if o['outcome'] == 1
            )
            
            conversion_rate = (
                n_positive_outcomes / n_predictions 
                if n_predictions > 0 else 0
            )
            
            stats[model_name] = {
                'predictions': n_predictions,
                'positive_outcomes': n_positive_outcomes,
                'conversion_rate': conversion_rate
            }
        
        # Calculate statistical significance
        from scipy.stats import chi2_contingency
        
        contingency_table = [
            [stats['A']['positive_outcomes'], 
             stats['A']['predictions'] - stats['A']['positive_outcomes']],
            [stats['B']['positive_outcomes'], 
             stats['B']['predictions'] - stats['B']['positive_outcomes']]
        ]
        
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        stats['test_results'] = {
            'p_value': p_value,
            'statistically_significant': p_value < 0.05,
            'winner': 'B' if stats['B']['conversion_rate'] > stats['A']['conversion_rate'] else 'A'
        }
        
        return stats

# Flask integration
ab_controller = ABTestController(model_a, model_b, split_ratio=0.5)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    user_id = data.get('user_id')
    features = data.get('features')
    
    prediction, model_used = ab_controller.predict(features, user_id)
    
    return jsonify({
        'prediction': prediction,
        'model': model_used,  # Can be hidden from user
        'request_id': str(uuid.uuid4())
    })

@app.route('/feedback', methods=['POST'])
def feedback():
    """Record actual outcome"""
    data = request.get_json()
    model_name = data.get('model')
    user_id = data.get('user_id')
    outcome = data.get('outcome')  # 1 for positive, 0 for negative
    
    ab_controller.record_outcome(model_name, user_id, outcome)
    return jsonify({'status': 'recorded'})

@app.route('/ab_stats', methods=['GET'])
def ab_stats():
    """View A/B test results"""
    stats = ab_controller.get_statistics()
    return jsonify(stats)
```

---

**Q19: What are the challenges of deploying deep learning models vs traditional ML models?**

**A:**

| Challenge | Traditional ML | Deep Learning |
|-----------|----------------|---------------|
| **Model size** | Small (KB-MB) | Large (MB-GB) |
| **Inference time** | Fast (ms) | Slower (10-100ms+) |
| **Hardware requirements** | CPU sufficient | Often needs GPU |
| **Dependencies** | Minimal (numpy, sklearn) | Heavy (TensorFlow, PyTorch, CUDA) |
| **Versioning** | Simple (pickle file) | Complex (weights + architecture) |
| **Preprocessing** | Moderate | Extensive (normalization, augmentation) |
| **Latency sensitivity** | Good for real-time | Challenging for real-time |

**Specific challenges:**

**1. Model Size:**
```python
# Traditional ML
pickle.dump(sklearn_model, f)  # 5 MB

# Deep Learning
torch.save(pytorch_model.state_dict(), f)  # 500 MB
# Problem: Slow to load, expensive to transfer
```

**Solutions:**
- Model quantization (reduce precision: float32 → int8)
- Model pruning (remove unnecessary connections)
- Model distillation (train smaller model to mimic larger one)

```python
# PyTorch quantization
import torch.quantization

quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
# Can reduce size by 75% with minimal accuracy loss
```

**2. GPU Requirements:**
```python
# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Problem: Production servers may not have GPUs
# Solution: Model serving platforms (TensorFlow Serving, TorchServe)
```

**3. Complex Preprocessing:**
```python
# Must save preprocessing pipeline with model
preprocessing_pipeline = {
    'resize': (224, 224),
    'normalize_mean': [0.485, 0.456, 0.406],
    'normalize_std': [0.229, 0.224, 0.225],
    'augmentations': ['horizontal_flip', 'rotation']
}

# Save with model
torch.save({
    'model_state': model.state_dict(),
    'preprocessing': preprocessing_pipeline
}, 'model_complete.pth')
```

**4. Batching for efficiency:**
```python
# Inefficient: Process one at a time
for sample in samples:
    prediction = model(sample)  # GPU underutilized

# Efficient: Batch processing
batch = torch.stack(samples)
predictions = model(batch)  # Full GPU utilization
```

---

**Q20: How do you implement model monitoring and logging in production?**

**A:** Comprehensive monitoring is critical for production ML systems.

```python
# monitoring.py
import logging
import time
from datetime import datetime
from functools import wraps
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MLMonitor:
    def __init__(self):
        self.prediction_count = 0
        self.error_count = 0
        self.latencies = []
        self.feature_stats = []
        
    def log_prediction(self, features, prediction, latency, error=None):
        """Log prediction details"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'features': features.tolist() if hasattr(features, 'tolist') else features,
            'prediction': prediction,
            'latency_ms': latency * 1000,
            'error': str(error) if error else None
        }
        
        if error:
            self.error_count += 1
            logger.error(f"Prediction error: {json.dumps(log_entry)}")
        else:
            self.prediction_count += 1
            self.latencies.append(latency)
            logger.info(f"Prediction: {json.dumps(log_entry)}")
        
        # Store for analysis
        self.feature_stats.append(features)
        
    def get_metrics(self):
        """Get monitoring metrics"""
        if not self.latencies:
            return {}
        
        return {
            'total_predictions': self.prediction_count,
            'total_errors': self.error_count,
            'error_rate': self.error_count / (self.prediction_count + self.error_count),
            'avg_latency_ms': sum(self.latencies) / len(self.latencies) * 1000,
            'p50_latency_ms': self.percentile(self.latencies, 0.5) * 1000,
            'p95_latency_ms': self.percentile(self.latencies, 0.95) * 1000,
            'p99_latency_ms': self.percentile(self.latencies, 0.99) * 1000
        }
    
    @staticmethod
    def percentile(data, percentile):
        import numpy as np
        return np.percentile(data, percentile * 100)

# Global monitor
monitor = MLMonitor()

def monitor_prediction(f):
    """Decorator to monitor predictions"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        error = None
        result = None
        
        try:
            result = f(*args, **kwargs)
        except Exception as e:
            error = e
            raise
        finally:
            latency = time.time() - start_time
            
            # Extract features from args/kwargs
            features = kwargs.get('features', args[0] if args else None)
            
            monitor.log_prediction(
                features=features,
                prediction=result,
                latency=latency,
                error=error
            )
        
        return result
    return wrapper

# Flask integration
@app.route('/predict', methods=['POST'])
@monitor_prediction
def predict():
    data = request.get_json()
    features = preprocess(data)
    prediction = model.predict(features)
    return jsonify({'prediction': prediction})

@app.route('/metrics', methods=['GET'])
def metrics():
    """Endpoint to view metrics"""
    return jsonify(monitor.get_metrics())

# Advanced: Integration with Prometheus
from prometheus_client import Counter, Histogram, generate_latest

prediction_counter = Counter('predictions_total', 'Total predictions')
error_counter = Counter('prediction_errors_total', 'Total prediction errors')
latency_histogram = Histogram('prediction_latency_seconds', 'Prediction latency')

@app.route('/predict', methods=['POST'])
def predict():
    with latency_histogram.time():
        try:
            # Prediction logic
            prediction_counter.inc()
            return jsonify({'prediction': result})
        except Exception as e:
            error_counter.inc()
            raise

@app.route('/prometheus_metrics')
def prometheus_metrics():
    return generate_latest()
```

---

## 8. Real-World Project Examples

### Project 1: House Price Prediction API

**Complete implementation:**

```python
# train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load data
df = pd.read_csv('housing_data.csv')

# Feature engineering
df['price_per_sqft'] = df['price'] / df['sqft']
df['age'] = 2025 - df['year_built']
df['has_garage'] = df['garage_size'] > 0

# Prepare features
feature_cols = ['sqft', 'bedrooms', 'bathrooms', 'age', 'price_per_sqft', 'has_garage']
X = df[feature_cols]
y = df['price']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print(f"R²: {r2_score(y_test, y_pred):.3f}")
print(f"RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred)):,.0f}")

# Save
pickle.dump(model, open('house_price_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump(feature_cols, open('feature_names.pkl', 'wb'))
```

```python
# app.py
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load artifacts
model = pickle.load(open('house_price_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
feature_names = pickle.load(open('feature_names.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extract and validate features
        features = []
        for feature in feature_names:
            if feature not in data:
                return jsonify({'error': f'Missing feature: {feature}'}), 400
            features.append(float(data[feature]))
        
        # Prepare input
        features_array = np.array([features])
        features_scaled = scaler.transform(features_array)
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        
        # Get prediction interval (using standard deviation of predictions)
        predictions_all = [tree.predict(features_scaled)[0] 
                          for tree in model.estimators_]
        std = np.std(predictions_all)
        
        return jsonify({
            'predicted_price': float(prediction),
            'confidence_interval': {
                'lower': float(prediction - 1.96 * std),
                'upper': float(prediction + 1.96 * std)
            },
            'currency': 'USD'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

### Project 2: Image Classification with PyTorch + Flask

```python
# train_cnn.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Data augmentation and normalization
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# Load data
train_dataset = datasets.ImageFolder('data/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Use pretrained ResNet
model = models.resnet18(pretrained=True)
num_classes = len(train_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Save
torch.save({
    'model_state': model.state_dict(),
    'class_names': train_dataset.classes,
    'transform': transform
}, 'image_classifier.pth')
```

```python
# app.py
from flask import Flask, request, jsonify
import torch
from torchvision import models, transforms
from PIL import Image
import io

app = Flask(__name__)

# Load model
checkpoint = torch.load('image_classifier.pth', map_location='cpu')
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, len(checkpoint['class_names']))
model.load_state_dict(checkpoint['model_state'])
model.eval()

class_names = checkpoint['class_names']
transform = checkpoint['transform']

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    # Read image
    image_file = request.files['image']
    image_bytes = image_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Preprocess
    image_tensor = transform(image).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
    
    return jsonify({
        'predicted_class': class_names[predicted_class],
        'confidence': float(probabilities[predicted_class]),
        'all_probabilities': {
            class_names[i]: float(probabilities[i])
            for i in range(len(class_names))
        }
    })

if __name__ == '__main__':
    app.run(debug=True)
```

---

## 9. Best Practices & Tips

### Development Best Practices

1. **Always split your data properly**
   ```python
   # Wrong: Using all data for training
   model.fit(X, y)
   
   # Right: Train-validation-test split
   X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3)
   X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)
   ```

2. **Use cross-validation**
   ```python
   from sklearn.model_selection import cross_val_score
   scores = cross_val_score(model, X_train, y_train, cv=5)
   print(f"CV Score: {scores.mean():.3f} (+/- {scores.std():.3f})")
   ```

3. **Version everything**
   ```python
   model_metadata = {
       'version': '2.0.0',
       'training_date': '2025-01-15',
       'features': feature_names,
       'metrics': {'accuracy': 0.95, 'f1': 0.93},
       'hyperparameters': {'n_estimators': 100, 'max_depth': 20}
   }
   ```

4. **Handle class imbalance**
   ```python
   from sklearn.utils.class_weight import compute_class_weight
   
   class_weights = compute_class_weight('balanced', 
                                         classes=np.unique(y_train),
                                         y=y_train)
   model = RandomForestClassifier(class_weight='balanced')
   ```

5. **Monitor for data leakage**
   ```python
   # Wrong: Fit scaler on all data
   scaler.fit(X)  # Test data leaked!
   
   # Right: Fit only on training data
   scaler.fit(X_train)
   X_test_scaled = scaler.transform(X_test)
   ```
