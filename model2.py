import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# --- 1. Configuration ---
NUM_CLIENTS = 5
BATCH_SIZE = 32
NUM_ROUNDS = 20
CLIENT_LEARNING_RATE = 0.001
SERVER_LEARNING_RATE = 0.001
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --- 2. Load and Preprocess Data ---
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
    client_ids = [f'client_{i}' for i in range(NUM_CLIENTS)]
    
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
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=len(x))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda x, y: (x, tf.one_hot(y, depth=3)))
    return dataset

# --- 4. Define the Model Architecture ---
def create_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CLIENT_LEARNING_RATE),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()]
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
    print("Loading and preprocessing data...")
    X, y, num_classes, scaler, label_encoders, target_encoder, feature_columns = load_and_preprocess_data()
    
    # Create client data
    print("Creating client data partitions...")
    client_data = create_client_data(X, y)
    
    # Create global model
    input_shape = (X.shape[1],)
    global_model = create_model(input_shape, num_classes)
    initial_weights = global_model.get_weights()
    
    # Training loop
    print("\nStarting Federated Training...")
    for round_num in range(NUM_ROUNDS):
        print(f"\nRound {round_num + 1}/{NUM_ROUNDS}")
        
        # Initialize list to store client weights and metrics
        client_weights = []
        client_metrics = []
        
        # Client training phase
        for client_id in client_data.keys():
            # Create client model with current global weights
            client_model = create_model(input_shape, num_classes)
            client_model.set_weights(global_model.get_weights())
            
            # Train on client data
            metrics, weights = train_client_model(client_model, client_data[client_id])
            client_weights.append(weights)
            client_metrics.append(metrics)
        
        # Aggregate metrics
        round_loss = np.mean([m['loss'][0] for m in client_metrics])
        round_accuracy = np.mean([m['categorical_accuracy'][0] for m in client_metrics])
        print(f"Average client metrics - Loss: {round_loss:.4f}, Accuracy: {round_accuracy:.4f}")
        
        # Server aggregation phase
        global_weights = aggregate_weights(client_weights)
        global_model.set_weights(global_weights)
    
    print("\nFederated Training Complete!")
    
    # Evaluate global model on each client
    print("\nEvaluating Model on Individual Clients:")
    for client_id, (client_x, client_y) in client_data.items():
        client_dataset = create_dataset(client_x, client_y)
        loss, accuracy = global_model.evaluate(client_dataset, verbose=0)
        print(f"{client_id} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
    
    # Save the model
    global_model.save_weights('federated_credit_model_weights.h5')
    print("\nTrained model weights saved to 'federated_credit_model_weights.h5'")

if __name__ == "__main__":
    main()