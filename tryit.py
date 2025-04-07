import streamlit as st
import joblib  # Assuming the model is saved using joblib

# Load the model
model = joblib.load("path_to_your_model.pkl")  # Replace with the actual path to your model

# Streamlit app
st.title("Lightweight Model App")
st.write("Interact with your model below:")

# Input section
user_input = st.text_input("Enter input for the model:")

# Prediction section
if st.button("Run Model"):
    # Replace with your model's prediction logic
    if user_input:
        predictions = model.predict([user_input])  # Adjust input format as needed
        if isinstance(predictions, (list, tuple)):
            st.write("Predictions:")
            for idx, prediction in enumerate(predictions, start=1):
                st.write(f"{idx}. {prediction}")
        else:
            st.write("Prediction:", predictions)
    else:
        st.write("Please enter valid input.")
