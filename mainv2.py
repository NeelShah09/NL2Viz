import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import json

from langgraph.graph import StateGraph, END, START
from langchain_huggingface import HuggingFaceEndpoint
import google.generativeai as genai
from typing import TypedDict
from dotenv import load_dotenv

class GraphState(TypedDict):
    user_input: str
    model_name: str
    file_path: str
    data: pd.DataFrame
    response: str
    visualization_code: str

class DataVisualizer:
    def __init__(self, data_path: str):
        """Initialize the DataVisualizer with model name and file path."""
        if not os.path.exists('config.json'):
            raise FileNotFoundError("Configuration file 'config.json' not found.")
        self._config = json.load(open('config.json'))

        self._model_name = self._config.get("LLM").get("MODEL_ID")
        if not self._model_name:
            raise ValueError("Model name is not specified in the configuration file!")
        self._model = self._get_model()
        self._data_path = data_path
        self._data = self._load_data()
        self._user_input = None

    def _load_data(self):
        """Function to load data from a CSV file."""
        try:
            return pd.read_csv(self._data_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self._data_path}")

    def _get_model(self):
        """Function to get the HuggingFace model endpoint."""
        try:
            if self._config.get("LLM").get("MODEL_TYPE") == "HuggingFace":
                token = os.getenv("HUGGINGFACE_TOKEN")
                if not token:
                    raise ValueError("HuggingFace API token is not set in the environment variables.")
                return HuggingFaceEndpoint(
                    repo_id=self._model_name,
                    temperature = self._config.get("LLM").get("TEMPERATURE",0.1),
                    max_new_tokens=self._config.get("LLM").get("MAX_NEW_TOKENS",128),
                    huggingfacehub_api_token=token
                )
            elif self._config.get("LLM").get("MODEL_TYPE") == "Gemini":
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                return genai.GenerativeModel(self._model_name)
            else:
                raise ValueError("Unsupported model type specified in the configuration file!")
        except Exception as e:
            raise RuntimeError(f"Error initializing model: {e}")

    def _create_prompt_template(self):
        """Function to create a prompt template for the model."""
        return f"""
You are an expert in Data Visualization in Python specifically using Pandas, Seaborn and Matplotlib.
You will be given questions related to data visualization and you generate Python code for creating that visualization in python.
Your job is to right syntactically correct code with no commented explanation.
Note:
    - DO NOT IMPORT ANY LIBRARIES AND DATASETS.
    - DATA IS ALREADY LOADED IN A VARIABLE CALLED `df` WHICH IS A PANDAS DATAFRAME.
    - THE DATAFRAME CONTAINS THE COLUMNS: {', '.join(self._data.columns)}
    - USE APPROPRIATE OPERATIONS ON THE DATAFRAME TO CREATE THE VISUALIZATION AS PER USER INPUT

    - SAMPLE DATA:
    {self._data.head().to_string(index=False)}

User Query: {self._user_input}

Python Code:
"""
    
    def _process_response(self, response: str) -> str:
        """Function to preprocess the response to extract the Python code. Extracts Python code from the response."""
        response = re.split("```python|```Python|```", response.strip())
        response = [x for x in response if re.search(r"```", x) is None and x != '']
        return response[0].strip()

    def _generate_code(self, state: GraphState):
        """Function to generate code based on user input and data."""
        prompt_template = self._create_prompt_template()
        try:
            if self._config.get("LLM").get("MODEL_TYPE") == "HuggingFace":
                model_response = self._model.invoke(prompt_template, stop=["User:"])
                self._visualization_code = self._process_response(model_response)
            elif self._config.get("LLM").get("MODEL_TYPE") == "Gemini":
                model_response = self._model.generate_content(prompt_template)
                self._visualization_code = self._process_response(model_response.text)
            return {**state, "data": self._data, "response": model_response, "visualization_code": self._visualization_code}
        except Exception as e:
            raise RuntimeError(f"Error generating code: {e}")
        
    def _execute_code(self, state: GraphState):
        """Function to execute the generated code."""
        try:
            exec(self._visualization_code, {'df': self._data, 'plt': plt, 'sns': sns})
            if self._visualization_code.strip().split()[-1] != "plt.show()":
                plt.show()
            return state
        except Exception as e:
            raise RuntimeError(f"Error executing the code: {e}")
        
    def build_graph(self):
        """Function to create a state graph for generating and executing visualization code."""
        
        # Create the state graph
        builder = StateGraph(GraphState)
        builder.add_node("Generate Code", self._generate_code)
        builder.add_node("Execute Code", self._execute_code)
        
        # Add the start, end, and transition edges
        builder.add_edge(START, "Generate Code")
        builder.add_edge("Generate Code", "Execute Code")
        builder.add_edge("Execute Code", END)
        
        self._graph = builder.compile()

    def run(self, user_input: str):
        """Function to run the visualization process."""
        self._user_input = user_input
        self.build_graph()
        # state = {"user_input": user_input, "model_name": self._model_name, "file_path": self._data_path}
        return self._graph.invoke({
            "user_input": user_input,
            "model_name": self._model_name,
            "file_path": self._data_path})

if __name__ == "__main__":
    load_dotenv(override=True)
    file_path = "data/exams.csv"
    visualizer = DataVisualizer(file_path)
    visualizer.build_graph()
    while True:
        user_input = input("Enter your query (or 'exit' to quit): ")
        if user_input.lower() in ['exit','quit','stop','close','q','Q']:
            break
        visualizer.run(user_input)