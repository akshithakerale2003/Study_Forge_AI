import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash-lite",
    temperature=0.4,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

def generate_notes(topic, level="Beginner"):
    """Generate study notes for a topic"""
    
    prompt = ChatPromptTemplate.from_template(
        """Create comprehensive study notes on "{topic}" for {level} level students.

Include:
1. **Introduction** - What is {topic}?
2. **Key Concepts** - Main ideas explained simply
3. **Important Points** - Bullet points for quick revision
4. **Examples** - Practical examples where applicable
5. **Summary** - Quick recap of what was learned
6. **Practice Questions** - 3 questions to test understanding

Make it engaging and easy to understand."""
    )
    
    chain = prompt | llm
    result = chain.invoke({"topic": topic, "level": level})
    
    return result.content 