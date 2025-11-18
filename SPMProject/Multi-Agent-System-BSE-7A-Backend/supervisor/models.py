import google.generativeai as genai
import os

genai.configure(api_key="AIzaSyCN1eqT2-h0UtGKTC6YKgtgsNWOsnF7aoA")

print("Available models:")
for model in genai.list_models():
    print(f"  - {model.name}")