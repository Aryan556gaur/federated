import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras import Sequential

# --- 1. Configuration ---
NUM_CLIENTS = 5
BATCH_SIZE = 32
NUM_ROUNDS = 2
CLIENT_LEARNING_RATE = 0.001
SERVER_LEARNING_RATE = 0.001
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- 2. Load and Preprocess Data ---
@st.cache_resource
def load_and_preprocess_data():
    # Load data
    train_data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')
    
    # Extract features and target
    feature_columns = [col for col in train_data.columns if col not in ['ID', 'Customer_ID', 'Credit_Score']]
    X = train_data[feature_columns]
    y = train_data['Credit_Score']
    
    # Preprocess categorical columns
    categorical_columns = X.select_dtypes(include=['object']).columns
    numerical_columns = X.select_dtypes(exclude=['object']).columns
    
    # Create preprocessing pipelines
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Scale numerical features
    scaler = StandardScaler()
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
    
    # Convert target to numeric
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)
    num_classes = len(target_encoder.classes_)
    
    # Convert to numpy arrays
    X = X.values.astype(np.float32)
    y = y.astype(np.int32)
    
    return X, y, num_classes, scaler, label_encoders, target_encoder, feature_columns

# --- 3. Partition Data into Clients ---
def create_client_data(X, y):
    # Simulate client data by partitioning the dataset
    client_data = {}
    client_ids = [f'client {i+1}' for i in range(NUM_CLIENTS)]
    
    # Get unique customers and their indices
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # Split indices among clients
    client_partitions = np.array_split(indices, NUM_CLIENTS)
    
    for i, client_id in enumerate(client_ids):
        client_indices = client_partitions[i]
        client_X = X[client_indices]
        client_y = y[client_indices]
        client_data[client_id] = (client_X, client_y)
    
    return client_data

def create_dataset(x, y, batch_size=BATCH_SIZE):
    ds = tf.data.Dataset.from_tensor_slices((x.astype('float32'), y.astype('int32')))
    ds = ds.shuffle(len(x), seed=SEED)
    ds = ds.batch(batch_size)
    return ds

# --- 4. Define the Model Architecture ---
def create_model(input_shape, num_classes):
    model = Sequential([
        tf.keras.layers.Input(shape=input_shape),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(512, activation='relu'),
        Dropout(0.3),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(256, activation='relu'),
        Dropout(0.3),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        Dense(32, activation='relu'),
        Dropout(0.1),
        
        Dense(3, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# --- 5. Federated Learning Functions ---
def train_client_model(client_model, client_data):
    """Train model on client data for one epoch."""
    x, y = client_data
    dataset = create_dataset(x, y)
    history = client_model.fit(dataset, epochs=1, verbose=0)
    return history.history, client_model.get_weights()

def aggregate_weights(weights_list):
    """Aggregate weights from multiple clients using FedAvg."""
    # Simple average of weights
    avg_weights = [np.zeros_like(w) for w in weights_list[0]]
    for weights in weights_list:
        for i, w in enumerate(weights):
            avg_weights[i] += w / len(weights_list)
    return avg_weights

# --- 6. Main Training Loop ---
def main():
    # Load and preprocess data
    with st.spinner("Loading and preprocessing data..."):
        # print("Loading and preprocessing data...")
        X, y, num_classes, scaler, label_encoders, target_encoder, feature_columns = load_and_preprocess_data()
        
        # Create client data
        print("Creating client data partitions...")
        client_data = create_client_data(X, y)
        
        # Create global model
        input_shape = (X.shape[1],)
    
    st.success("Data loaded successfully!")
    st.write(f"🔹 Number of Clients: {NUM_CLIENTS}")
    st.write(f"🔹 Classes: {num_classes}")
    st.write(f"🔹 Features: {input_shape[0]}")

    global_model = create_model(input_shape, num_classes)
    initial_weights = global_model.get_weights()
        
    client_weights = []
    client_metrics = []

    if st.button("Train for Client 1"):
        client_model = create_model(input_shape, num_classes)
        client_model.set_weights(global_model.get_weights())

        metrics, weights = train_client_model(client_model, client_data[list(client_data.keys())[0]])
        client_weights.append(weights)
        client_metrics.append(metrics)

        round_loss = np.mean([m['loss'][0] for m in client_metrics])
        round_accuracy = np.mean([m['accuracy'][0] for m in client_metrics])
        # print(f"Average client metrics - Loss: {round_loss:.4f}, Accuracy: {round_accuracy:.4f}")
        
        # Server aggregation phase
        global_weights = aggregate_weights(client_weights)
        global_model.set_weights(global_weights)

        st.write(f"🔹 Avg Loss: {round_loss:.4f} | Avg Accuracy: {round_accuracy:.4f}")

    if st.button("Train for Client 2"):
        client_model = create_model(input_shape, num_classes)
        client_model.set_weights(global_model.get_weights())

        metrics, weights = train_client_model(client_model, client_data[list(client_data.keys())[1]])
        client_weights.append(weights)
        client_metrics.append(metrics)

        round_loss = np.mean([m['loss'][0] for m in client_metrics])
        round_accuracy = np.mean([m['accuracy'][0] for m in client_metrics])
        # print(f"Average client metrics - Loss: {round_loss:.4f}, Accuracy: {round_accuracy:.4f}")
        
        # Server aggregation phase
        global_weights = aggregate_weights(client_weights)
        global_model.set_weights(global_weights)

        st.write(f"🔹 Avg Loss: {round_loss:.4f} | Avg Accuracy: {round_accuracy:.4f}")

    if st.button("Train for Client 3"):
        client_model = create_model(input_shape, num_classes)
        client_model.set_weights(global_model.get_weights())

        metrics, weights = train_client_model(client_model, client_data[list(client_data.keys())[2]])
        client_weights.append(weights)
        client_metrics.append(metrics)

        round_loss = np.mean([m['loss'][0] for m in client_metrics])
        round_accuracy = np.mean([m['accuracy'][0] for m in client_metrics])
        # print(f"Average client metrics - Loss: {round_loss:.4f}, Accuracy: {round_accuracy:.4f}")
        
        # Server aggregation phase
        global_weights = aggregate_weights(client_weights)
        global_model.set_weights(global_weights)

        st.write(f"🔹 Avg Loss: {round_loss:.4f} | Avg Accuracy: {round_accuracy:.4f}")

    if st.button("Train for Client 4"):
        client_model = create_model(input_shape, num_classes)
        client_model.set_weights(global_model.get_weights())

        metrics, weights = train_client_model(client_model, client_data[list(client_data.keys())[3]])
        client_weights.append(weights)
        client_metrics.append(metrics)

        round_loss = np.mean([m['loss'][0] for m in client_metrics])
        round_accuracy = np.mean([m['accuracy'][0] for m in client_metrics])
        # print(f"Average client metrics - Loss: {round_loss:.4f}, Accuracy: {round_accuracy:.4f}")
        
        # Server aggregation phase
        global_weights = aggregate_weights(client_weights)
        global_model.set_weights(global_weights)

        st.write(f"🔹 Avg Loss: {round_loss:.4f} | Avg Accuracy: {round_accuracy:.4f}")

    if st.button("Train for Client 5"):
        client_model = create_model(input_shape, num_classes)
        client_model.set_weights(global_model.get_weights())

        metrics, weights = train_client_model(client_model, client_data[list(client_data.keys())[4]])
        client_weights.append(weights)
        client_metrics.append(metrics)

            
        # Aggregate metrics
        round_loss = np.mean([m['loss'][0] for m in client_metrics])
        round_accuracy = np.mean([m['accuracy'][0] for m in client_metrics])
        # print(f"Average client metrics - Loss: {round_loss:.4f}, Accuracy: {round_accuracy:.4f}")
        
        # Server aggregation phase
        global_weights = aggregate_weights(client_weights)
        global_model.set_weights(global_weights)

        st.write(f"🔹 Avg Loss: {round_loss:.4f} | Avg Accuracy: {round_accuracy:.4f}")
            
        st.success("✅ Training complete!")
        # print("\nFederated Training Complete!")

        global_model.save_weights('weights/federated_credit_model_weights.weights.h5')
        st.success("💾 Trained model weights saved to 'weights/federated_credit_model_weights.weights.h5")

    
    # Evaluate global model on each client
    if st.button("### 🧪 Evaluating on Clients"):
        global_model.load_weights('weights/federated_credit_model_weights.weights.h5')
        # print("\nEvaluating Model on Individual Clients:")
        for client_id, (client_x, client_y) in client_data.items():
            client_dataset = create_dataset(client_x, client_y)
            loss, accuracy = global_model.evaluate(client_dataset, verbose=0)
            
            st.write(f"{client_id} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
            # print(f"{client_id} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Save the model

if __name__ == "__main__":
    main()