!pip install transformers datasets torch scikit-learn flask streamlit

from transformers import pipeline

# Load a pre-trained model for toxicity detection
toxicity_pipeline = pipeline("text-classification", model="unitary/toxic-bert")

# Test the model
test_sentence = "I hate this product!"
result = toxicity_pipeline(test_sentence)
print(result)

def chatbot_response(user_input):
    # Analyze the input for toxicity
    toxicity_result = toxicity_pipeline(user_input)[0]
    toxicity_score = toxicity_result['score']
    toxic_label = toxicity_result['label']

    # Generate response
    if toxic_label == "TOXIC" and toxicity_score > 0.7:
        return "Please use kind words. I detect toxicity in your message."
    else:
        return "Thank you for your input! How can I assist you further?"
import streamlit as st

st.title("Toxicity-Detecting Chatbot")

user_input = st.text_input("You: ")
if user_input:
    response = chatbot_response(user_input)
    st.write(f"Bot: {response}")
