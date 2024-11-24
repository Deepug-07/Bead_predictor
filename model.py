import numpy as np
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Define your model architecture
def create_model(input_shape):
    model = Sequential()
    model.add(Dense(128, input_shape=(input_shape,), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(16, activation='relu'))
    model.add(Dense(2, activation='linear'))  # Output layer for bead height and width
    return model


def predict_outputs(wfs, ts, voltage):
    # Create a 2D array with the input parameters
    input_data = np.array([[wfs, ts, voltage]])

    input_shape = 3  # Adjust based on your input features
    model = create_model(input_shape)

    data=pd.read_csv('Dataset.csv')
    X=data[['WFS','TS','Voltage']].values
    y=data[['BH','BW']].values
    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)
    
    model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])
    model.load_weights('neural_network_weights_v2.weights.h5')
    
    
    input_data_scaled = scaler.transform(input_data)
    
    # Use the model to predict the outputs
    predictions = model.predict(input_data_scaled)
    
    # Return the predicted outputs
    return predictions[0]