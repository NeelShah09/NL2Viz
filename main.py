import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

from langgraph.graph import StateGraph, END, START
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from typing import TypedDict
from dotenv import load_dotenv

def load_data_from(file_path):
    """Function to load data from a CSV file."""
    return pd.read_csv(file_path)


def get_model(repo_id="meta-llama/Llama-2-7b-hf"):
    """Function to get the HuggingFace model endpoint."""
    return HuggingFaceEndpoint(
        repo_id=repo_id,
        temperature = 0.1,
        max_new_tokens=128,
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_TOKEN")
    )


def create_prompt_template(user_input: str, data: pd.DataFrame):
    """Function to create a prompt template for the model."""
    return f"""
You are an expert in Data Visualization in Python specifically using Pandas, Seaborn and Matplotlib.
You will be given questions related to data visualization and you generate Python code for creating that visualization in python.
Your job is to right syntactically correct code with no commented explanation.
Note:
    - DO NOT IMPORT ANY LIBRARIES AND DATASETS.
    - DATA IS ALREADY LOADED IN A VARIABLE CALLED `df` WHICH IS A PANDAS DATAFRAME.
    - THE DATAFRAME CONTAINS THE COLUMNS: {', '.join(data.columns)}
    - USE APPROPRIATE OPERATIONS ON THE DATAFRAME TO CREATE THE VISUALIZATION AS PER USER INPUT

    - SAMPLE DATA:
    {data.head().to_string(index=False)}

User Query: {user_input}

Python Code:
"""


def visualizer(user_input: str, model_name: str, file_path: str):
    """Function to create a state graph for generating and executing visualization code."""
    def generate_code(state: dict):
        """Function to generate code based on user input and data."""
        user_input = state['user_input']
        data = load_data_from(state['file_path'])
        state['data'] = data
        model = get_model(repo_id = state['model_name'])

        prompt_template = create_prompt_template(user_input, data)
        response = model.invoke(prompt_template, stop=["User:"])
        state['response'] = response

        return state

    def preprocess(response: str) -> str:
        """Function to preprocess the response to extract the Python code. Extracts Python code from the response."""
        response = re.split("```python|```Python|```", response.strip())
        response = [x for x in response if re.search(r"```", x) is None and x != '']
        return response[0].strip()

    
    def execute_code(state: dict):
        """Function to execute the generated code."""
        response = state['response']
        try:
            code = preprocess(response)
            print(code)
            exec(code, {'df': state['data'], 'plt': plt, 'sns': sns})
            if code.strip().split()[-1] != "plt.show()":
                plt.show()
        except Exception as e:
            raise RuntimeError(f"Error executing the code: {e}")

        state['execution_result'] = "Code executed successfully."
        return state

    class GraphState(TypedDict):
        """TypedDict to define the state of the graph."""
        user_input: str
        model_name: str
        file_path: str
        data: pd.DataFrame
        response: str
        execution_result: str
    
    # Create the state graph
    builder = StateGraph(GraphState)
    builder.add_node("Generate Code", generate_code)
    builder.add_node("Execute Code", execute_code)
    # Add the start, end, and transition edges
    builder.add_edge(START, "Generate Code")
    builder.add_edge("Generate Code", "Execute Code")
    builder.add_edge("Execute Code", END)
    
    return builder.compile()

if __name__ == "__main__":
    load_dotenv()
    
    model = "meta-llama/Llama-3.1-8B-Instruct"
    file_path = "data/exams.csv"

    while True:
        user_input = input("Enter your query (or 'exit' to quit): ")
        if user_input.lower() in ['exit','quit','stop','close','q','Q']:
            break
        visualizer(user_input, model, file_path).invoke({"user_input": user_input,"model_name": model,"file_path": file_path})