pip install matplotlib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, InputLayer
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Unique Title for the Dashboard
st.title("🤖 Interactive ANN Prediction & Visualization Dashboard")

# Upload dataset section
st.markdown("### 📂 Upload Your CSV Dataset")
uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

if uploaded_file is not None:
    # Load and preview dataset
    data = pd.read_csv(uploaded_file)
    st.markdown("#### 🔍 Dataset Preview")
    st.dataframe(data.head())
    st.markdown("#### 📊 Dataset Summary")
    st.write(data.describe())
    
    # Select target column and display class distribution
    target_col = st.selectbox("🎯 Select Target Column", data.columns)
    feature_cols = [col for col in data.columns if col != target_col]
    
    st.markdown("#### 📈 Target Column Distribution")
    fig_dist, ax_dist = plt.subplots()
    data[target_col].value_counts().plot(kind="bar", ax=ax_dist)
    ax_dist.set_xlabel("Classes")
    ax_dist.set_ylabel("Frequency")
    st.pyplot(fig_dist)
    
    # Data Preprocessing
    X = data[feature_cols].copy()
    y = data[target_col].copy()
    
    # Encode categorical features if needed
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col])
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)
    
    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Sidebar: Choose test size ratio
    test_size = st.sidebar.slider("Test Data Ratio", 0.1, 0.4, 0.2, step=0.05)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=2224)
    
    # Sidebar: Model Hyperparameters
    st.sidebar.markdown("### ⚙️ Model Hyperparameters")
    num_layers = st.sidebar.slider("Number of Hidden Layers", 1, 5, 2)
    neurons_per_layer = []
    activation_functions = []
    for i in range(num_layers):
        neurons = st.sidebar.slider(f"Neurons in Hidden Layer {i+1}", 5, 150, 50, step=5)
        activation = st.sidebar.selectbox(f"Activation Function for Layer {i+1}", ['relu', 'tanh', 'sigmoid'], key=f"act_{i}")
        neurons_per_layer.append(neurons)
        activation_functions.append(activation)
    
    optimizer_choice = st.sidebar.selectbox("Optimizer", ['adam', 'sgd', 'rmsprop'])
    lr = st.sidebar.number_input("Learning Rate", value=0.001, min_value=0.0001, max_value=0.1, step=0.0001, format="%.4f")
    loss_func = st.sidebar.selectbox("Loss Function", ['binary_crossentropy', 'mean_squared_error', 'hinge'])
    dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.2)
    epochs = st.sidebar.slider("Epochs", 10, 200, 50, step=10)
    batch_size = st.sidebar.slider("Batch Size", 8, 128, 32, step=8)
    early_stopping = st.sidebar.checkbox("Enable Early Stopping", value=False)
    
    # Train Model Button
    if st.button("🚀 Train the ANN Model"):
        # Build the ANN model
        model = Sequential()
        model.add(InputLayer(input_shape=(X_train.shape[1],)))
        for i in range(num_layers):
            model.add(Dense(units=neurons_per_layer[i], activation=activation_functions[i]))
            model.add(Dropout(rate=dropout_rate))
        model.add(Dense(1, activation='sigmoid'))
        
        # Select optimizer
        if optimizer_choice == "adam":
            optimizer_obj = Adam(learning_rate=lr)
        elif optimizer_choice == "sgd":
            optimizer_obj = SGD(learning_rate=lr)
        else:
            optimizer_obj = RMSprop(learning_rate=lr)
            
        model.compile(loss=loss_func, optimizer=optimizer_obj, metrics=['accuracy'])
        
        st.markdown("#### 🔧 Model Configuration")
        st.write("Optimizer:", optimizer_choice)
        st.write("Learning Rate:", lr)
        st.write("Neurons per Layer:", neurons_per_layer)
        st.write("Dropout Rate:", dropout_rate)
        st.write("Epochs:", epochs)
        st.write("Batch Size:", batch_size)
        
        # Optional early stopping callback
        callbacks = []
        if early_stopping:
            early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
            callbacks.append(early_stop)
        
        # Train the model
        with st.spinner('Training the model... Please wait!'):
            history = model.fit(X_train, y_train,
                                validation_data=(X_test, y_test),
                                epochs=epochs,
                                batch_size=batch_size,
                                callbacks=callbacks,
                                verbose=1)
        st.success("🎉 Model Training Completed!")
        
        # Plot Loss History
        fig_loss, ax_loss = plt.subplots()
        ax_loss.plot(history.history['loss'], label='Training Loss')
        ax_loss.plot(history.history['val_loss'], label='Validation Loss')
        ax_loss.set_xlabel("Epochs")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title("Loss over Epochs")
        ax_loss.legend()
        st.pyplot(fig_loss)
        
        # Plot Accuracy History
        fig_acc, ax_acc = plt.subplots()
        ax_acc.plot(history.history['accuracy'], label='Training Accuracy')
        ax_acc.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax_acc.set_xlabel("Epochs")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_title("Accuracy over Epochs")
        ax_acc.legend()
        st.pyplot(fig_acc)
        
        # Evaluate the model on test data
        st.markdown("### 📈 Model Evaluation on Test Data")
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        st.write(f"**Test Loss:** {test_loss:.4f}")
        st.write(f"**Test Accuracy:** {test_acc * 100:.2f}%")
        
        # Generate predictions and compute confusion matrix
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype("int32")
        cm = confusion_matrix(y_test, y_pred)
        st.markdown("#### 🤖 Confusion Matrix")
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        st.pyplot(fig_cm)
        
        # Display classification report
        st.markdown("#### 📋 Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
