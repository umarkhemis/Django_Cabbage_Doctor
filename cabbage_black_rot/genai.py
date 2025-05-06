
from transformers import pipeline
from rest_framework.exceptions import ValidationError
import google.generativeai as genai

# generator = pipeline("text-generation", model="gpt2")
# generator = pipeline("text-generation", model="gpt2")
# qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

genai.configure(api_key="AIzaSyB9IHWHbqggP__-hN9304vrJqTnvTDha3c")

qa_pipeline = genai.GenerativeModel("gemini-1.5-flash")

def get_disease_insight(disease_name):
    prompt = f"""
    For the crop disease called "{disease_name}" in Uganda, provide the following information in a clearly labeled format:

    Cause:
    [List the main causes specific to Uganda]

    Treatment:
    [Give treatments used by Ugandan farmers or commonly available in Uganda]

    Prevention:
    [How can this disease be prevented, particularly in Ugandan agricultural conditions]

    Return ONLY these 3 sections.
    """
    
    response = qa_pipeline.generate_content(prompt)
    return response.text