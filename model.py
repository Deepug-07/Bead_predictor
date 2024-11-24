import numpy as np
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.preprocessing import StandardScaler

# Define your model architecture
def create_model():
    model = Sequential()
    model.add(Dense(128, input_shape=(X_train.shape[1],), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(16, activation='relu'))

    model.add(Dense(2, activation='linear'))
    return model

# Load the model and weights
model = create_model()
model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])
model.load_weights('neural_network_weights_v2.weights.h5')


def predict_outputs(wfs, ts, voltage):
    # Create a 2D array with the input parameters
    input_data = np.array([[wfs, ts, voltage]])
    
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Use the model to predict the outputs
    predictions = model.predict(input_data_scaled)
    
    # Return the predicted outputs
    return predictions[0]