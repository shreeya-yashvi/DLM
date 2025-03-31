# Diabetes Prediction Using Artificial Neural Networks (ANN)  
🔗 **For the Streamlit Dashboard Link:** [Link to Dashboard](https://dlmprojects-awzttghmyh6xdcrs75xuzq.streamlit.app/)

---

## 🏆 Project Details
This project focuses on predicting the likelihood of diabetes using a deep learning model built with **Artificial Neural Networks (ANN)**. The model is trained on a structured dataset containing patient health records, including attributes such as glucose levels, BMI, and insulin. Hyperparameter tuning was emphasized to optimize model performance. The interactive dashboard allows users to modify hyperparameters and observe real-time results.

---

## 👥 Contributors
- **Shreeya Yashvi** – 055045 
- **Kashish Srivastava** – 055046

---

## 🔑 Key Activities
1. **Data Preprocessing:**  
   - Handling missing values.  
   - Encoding categorical variables (if applicable).  
   - Scaling numerical features for normalization.  

2. **Model Development:**  
   - Constructing an ANN with tunable hyperparameters using Streamlit.  
   - Allowing flexibility in the number of layers, neurons, and activation functions.  

3. **Hyperparameter Tuning:**  
   - Exploring different combinations of hidden layers, dropout rates, optimizers, and learning rates.  

4. **Visualization & Insights:**  
   - Real-time plots of training/validation loss and accuracy over epochs.  
   - Displaying model evaluation metrics such as confusion matrix and classification reports.  

5. **Managerial Interpretation:**  
   - Drawing actionable insights for clinical decision-making and healthcare optimization.  

---

## 💻 Technologies Used
- **Python:** Core programming language for data processing and model development.  
- **TensorFlow & Keras:** Deep learning framework for building and training ANN models.  
- **Pandas & NumPy:** Data manipulation and numerical computations.  
- **Matplotlib & Seaborn:** Data visualization and model performance analysis.  
- **Scikit-learn:** Preprocessing, train-test split, and evaluation metrics.  
- **Streamlit:** Building the interactive and user-friendly web interface.  

---

## 📊 Nature of Data
The dataset, `diabetes.csv`, consists of structured clinical data with a mix of continuous and binary variables. These attributes provide relevant information for predicting the risk of diabetes.

### 📌 Variable Information
| Feature                   | Type         | Description                                          |
|---------------------------|--------------|------------------------------------------------------|
| Pregnancies                | Continuous   | Number of pregnancies                               |
| Glucose                    | Continuous   | Plasma glucose concentration (mg/dl)                |
| BloodPressure              | Continuous   | Diastolic blood pressure (mm Hg)                    |
| SkinThickness              | Continuous   | Triceps skinfold thickness (mm)                     |
| Insulin                    | Continuous   | 2-hour serum insulin (mu U/ml)                      |
| BMI                        | Continuous   | Body mass index (kg/m²)                             |
| DiabetesPedigreeFunction   | Continuous   | Diabetes pedigree function score                    |
| Age                        | Continuous   | Age of the patient (years)                          |
| Outcome                    | Binary       | 1 = Diabetes present, 0 = No diabetes               |

---

## 🎯 Problem Statements
- How can ANN models effectively classify patients at risk of diabetes?  
- What impact does hyperparameter tuning have on model accuracy?  
- Can the model achieve high accuracy without overfitting?

---

## 🏗️ Model Information

### Input Layer
- Accepts all numerical attributes after scaling.
- No categorical encoding required for this dataset.

### Hidden Layers
- **Number of Layers:** Configurable between 1 and 5.  
- **Neurons per Layer:** User-defined (range: 5 to 150 neurons).  
- **Activation Functions:** Options include ReLU, tanh, and sigmoid.  
- **Dropout Layers:** Dropout applied after hidden layers to prevent overfitting.

### Output Layer
- **Neurons:** 1  
- **Activation Function:** Sigmoid (suitable for binary classification tasks).  

### Optimizer Options
- **Adam:** Default and most effective in balancing speed and accuracy.  
- **SGD:** Slower and requires more epochs for convergence.  
- **RMSprop:** Useful but may lead to instability depending on learning rate.  

### Loss Function Options
- **Binary Cross-Entropy:** Ideal for binary classification.  
- **Mean Squared Error:** Alternative but not preferred for classification tasks.  
- **Hinge Loss:** Less suitable but available for experimentation.  

---

## 📉 Observations from Hyperparameter Tuning

### 1️⃣ Number of Hidden Layers
- **1–2 layers:** Moderate accuracy (~78%).  
- **3–4 layers:** Optimal accuracy (~82%) without overfitting.  
- **5+ layers:** Slight improvement but with increased computational cost and risk of overfitting.  

### 2️⃣ Neurons per Layer
- **10–50 neurons:** Stable and consistent training.  
- **>50 neurons:** Marginal improvement, but risk of overfitting increases.  

### 3️⃣ Activation Functions
- **ReLU:** Performed best in hidden layers, enabling faster convergence.  
- **Tanh:** Acceptable performance but slightly inferior to ReLU.  
- **Sigmoid:** Restricted to the output layer for binary classification.  

### 4️⃣ Optimizer Comparison
- **Adam:** Best performance, balancing speed and accuracy.  
- **SGD:** Slower, requires more epochs for convergence.  
- **RMSprop:** Works well but may be unstable with higher learning rates.  

### 5️⃣ Dropout Rate
- **0–0.2:** Best accuracy (~82%).  
- **0.3–0.5:** Reduces overfitting but slightly impacts model performance.  

### 6️⃣ Epochs
- **50 epochs:** Sufficient for model convergence.  
- **100+ epochs:** No significant improvement; overfitting may occur.  

---

## 📈 Managerial Insights

### 🔹 Healthcare Applications
- **Early Diagnosis:**  
   Timely identification of diabetes risk can facilitate preventive care and lifestyle modifications, potentially reducing complications.  

- **Automated Risk Assessment:**  
   AI-based predictions minimize human error, ensuring consistent and accurate diagnoses.  

### 🔹 Business Value
- **Operational Efficiency:**  
   Healthcare providers can reduce workload by automating the screening process and allocating resources more effectively.  

- **Cost Reduction:**  
   Early detection can prevent costly long-term complications, translating into financial savings for both patients and healthcare institutions.  

---

## 🚀 Streamlit Dashboard Implementation
The **Streamlit Dashboard** serves as the user interface, offering real-time visualization and control over ANN configurations.

### Dashboard Highlights
- **Dataset Upload & Preview:**  
   Upload CSV files and instantly view data previews and summary statistics.  

- **Target Distribution Visualization:**  
   Visualize class distributions to assess potential imbalance in target outcomes.  

- **Model Hyperparameter Configuration:**  
   Adjust key hyperparameters:
   - **Hidden Layers:** 1–5 layers  
   - **Neurons per Layer:** 5–150 neurons  
   - **Activation Functions:** ReLU, tanh, sigmoid  
   - **Optimizer & Learning Rate:** Adam, SGD, RMSprop  
   - **Dropout Rate:** Configurable to prevent overfitting  
   - **Epochs & Batch Size:** Adjustable for fine-tuning training behavior  

- **Model Training & Visualization:**  
   - Real-time loss and accuracy plots  
   - Evaluation of test data using confusion matrix and classification report  

---

## 📜 License
This project is **open-source** and welcomes further improvements.  
**Contributions and suggestions** can be made via pull requests or GitHub issues.

---

## 🤝 Contributions
Suggestions, improvements, and forks are welcome! Open a pull request or raise an issue. 💡
