import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Validate required environment variables
required_vars = [
    "OPENAI_API_KEY",
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_PROJECT"
]

missing = [var for var in required_vars if not os.getenv(var)]

if missing:
    raise EnvironmentError(
        f"Missing environment variables: {', '.join(missing)}"
    )

# Create model
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7
)

# Create prompt template
prompt = ChatPromptTemplate.from_template(
    "Answer the following question clearly:\n\n{question}"
)

# Build chain
chain = prompt | llm

# Input
question = "What are three benefits of learning Python?"

# Invoke chain
response = chain.invoke({
    "question": question
})

# Print response
print("\nModel Response:\n")
print(response.content)