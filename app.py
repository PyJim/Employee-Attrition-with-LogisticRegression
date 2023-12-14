from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

lr_model = joblib.load('employee_attrition.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        data = request.get_json()
        input_data = pd.DataFrame(data, index=[0])

        numeric_columns = input_data.select_dtypes(include=['int64', 'float64']).columns
        categorical_columns = input_data.select_dtypes(include=['object']).columns

        label_encoder = LabelEncoder()

        for col in categorical_columns:
            input_data[col] = label_encoder.fit_transform(input_data[col])

        df_onehot = pd.get_dummies(input_data, columns=categorical_columns, drop_first=True)

        # Drop the original categorical columns after one-hot encoding
        input_data = input_data.drop(categorical_columns, axis=1)

        # Concatenate the one-hot encoded data with the original numeric columns
        input_data = pd.concat([input_data, df_onehot], axis=1)

        scaler = StandardScaler()
        input_data_scaled = scaler.fit_transform(input_data)
        
        # Make predictions using the trained model
        prediction = lr_model.predict(input_data_scaled)

        # Convert the prediction to a human-readable format using label encoder
        predicted_label = label_encoder.inverse_transform(prediction.astype(int))[0]

        # Prepare the response JSON
        response = {
            'prediction': predicted_label
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
