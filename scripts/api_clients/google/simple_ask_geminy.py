import google.generativeai as genai


# Initialize API and model outside of functions
GOOGLE_API_KEY = "AIzaSyCz013OoO4rIgksmBJYqad31twXoCZHQpE"  # Replace with your actual API key
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

def get_ai_response(prompt):
    response = model.generate_content(prompt)
    return response.text

if __name__ == "__main__":
    user_question = "How do I know if I have a mental health issue?"
    prompt = f"""
                You are an expert psychologist. You display care, love and knows how to listen.
                User's question: {user_question}
            """
    response = get_ai_response(prompt)
    print(response)