#Importing necessary libraries
# flask->creating flask application
# request->to handle incoming http requests
# jsonify->to create json requests
# render_template->to render html templates
# pickle->for loading pre-trained model and vectorizer
from flask import Flask, request, jsonify, render_template
import pickle

# Initialize the Flask application
app = Flask(__name__)

# Load the pre-trained logistic regression model and vectorizer from disk
with open('spam_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Define the home route that renders the index.html template
@app.route('/')
def home():
    return render_template('index.html')

# Define the /predict route to handle POST requests for spam prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Extract the email text from the request
    email_text = request.json['email_text']
    
    # Transform the email text into the vectorized form
    email_vector = vectorizer.transform([email_text])
    
    # Predict whether the email is spam (1) or not spam (0)
    prediction = model.predict(email_vector)[0]
    
    # Return the prediction result as a JSON response
    return jsonify({'prediction': 'spam' if prediction == 0 else 'not spam'})

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
