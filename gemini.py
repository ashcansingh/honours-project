import google.generativeai as genai

def generate_kg(text):
    # Set your API key
    genai.configure(api_key="AIzaSyB-qFRii0jjlQcag0cFtdCtUxzBtTEomKk")

    model = genai.GenerativeModel("gemini-2.0-flash")

    prompt = """
    You are tasked with extracting all triples from the text in the form of (entity) - (Relationship) - (entity). It is to be used in a knowledge graph.

    Text:\n"""

    response = model.generate_content(f"{prompt} + {text}")

    print(response.text)

