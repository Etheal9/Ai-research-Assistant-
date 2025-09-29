# function chinining
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.output_parsers import StrOutputParser

load_dotenv()

custemer_greeting_template = PromptTemplate(
    input_variables=["custemer_name", "product_name", "issue_description"],
    template=""" you are exprinced customer support specialist for {product_name}
    """
)

# first chain 
question_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Generate 3 questions about {topic}:"
)

# second chain

answer_prompt = PromptTemplate(
    input_variables=["questions"],
    template="Answer the following questions:\n{questions} your response must contain your answer and the question both."
)

llm = ChatGroq(tmeperature=0)

# creating a chain

output_parser = StrOutputParser()

question_chain = question_prompt | llm | output_parser

answer_chain = answer_prompt | llm | output_parser

def create_answer_input(output):
    return {"questions": output}

qa_chain = question_chain | create_answer_input | answer_chain

# this is the test questions
questions = question_chain.invoke({"topic": "artificial intellgent"})