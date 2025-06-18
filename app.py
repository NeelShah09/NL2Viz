import gradio as gr
from dotenv import load_dotenv
import json
import os
import time
from agents import DataVisualizer

def run_with_gradio(file_path: str, visualizer: DataVisualizer):
    """Function to run the Gradio interface for the DataVisualizer."""   
    def gradio_handler(user_query):
        result = visualizer.run(user_query)
        return result['plot_path'] if 'plot_path' in result else "No visualization generated."

    def close_app():
        time.sleep(1)
        os._exit(0)
    
    demo = gr.Interface(
        fn=gradio_handler,
        inputs=gr.Textbox(label="Enter your visualization query", placeholder="e.g., Show distribution of math score"),
        outputs=gr.Image(type="filepath",label="Generated Visualization"),
        title="LLM-based Data Visualizer",
        description="Enter a natural language query to generate Python visualizations using Seaborn/Matplotlib.",
        live=False
    )
    demo.launch()

if __name__ == "__main__":
    configuration = json.load(open('config.json'))
    if configuration["APP"]["MODE"] not in ["GUI", "CLI"]:
        raise ValueError("Invalid Mode of running. Set the config \"APP\":\"MODE\" appropriately to from [\"GUI\", \"CLI\"]")
    
    load_dotenv(override=True)
    file_path = "data/exams.csv"
    visualizer = DataVisualizer(file_path)
    visualizer.build_graph()

    if configuration["APP"]["MODE"] == "GUI":
        try:
            run_with_gradio(file_path="data/exams.csv", visualizer=visualizer)
        except KeyboardInterrupt as e:
            pass
    elif configuration["APP"]["MODE"] == "CLI":
        while True:
            user_input = input("Enter your query (or 'exit' to quit): ")
            if user_input.lower() in ['exit','quit','stop','close','q','Q']:
                break
            visualizer.run(user_input)