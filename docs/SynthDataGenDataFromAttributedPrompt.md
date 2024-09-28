# DataFromAttributedPrompt

## Overview
`DataFromAttributedPrompt` is a Python class designed to generate synthetic data based on a set of attributes and user instructions. It utilizes language models (LLMs) to generate prompts and retrieve responses that can be saved as a DataFrame or exported to an Excel file.

## Table of Contents
- [Installation](#installation)
- [Language Model Setup](#language-model-setup)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Contributing](#contributing)

## Installation
To use the `DataFromAttributedPrompt` class, you need to have Python 3.9+ installed. You can install the `indoxGen` package using pip:

```bash
pip install indoxGen
```

## Language Model Setup
`DataFromAttributedPrompt` requires an LLM (Language Model) for generating responses from provided prompts. The `indoxGen` library provides a unified interface for various language models. Here's how to set up the language model for this class:

```python
from indoxGen.llms import IndoxApi
import os
from dotenv import load_dotenv

load_dotenv()
INDOX_API_KEY = os.getenv("INDOX_API_KEY")

LLM = IndoxApi(api_key=INDOX_API_KEY)
```

The `indoxGen` library supports various models, including:
- OpenAI
- Mistral
- Ollama
- Google AI
- Hugging Face models

Additionally, `indoxGen` provides routing for OpenAI, enabling easy switching between different models.

## Usage
Here's a basic example of how to use the `DataFromAttributedPrompt` class:

```python
from indoxGen.synthCore import DataFromAttributedPrompt

# Define the arguments for generating prompts
args = {
    "instruction": "Generate a {adjective} sentence that is {length}.",
    "attributes": {
        "adjective": ["serious", "funny"],
        "length": ["short", "long"]
    },
    "llm": LLM
}

# Create an instance of DataFromAttributedPrompt
dataset = DataFromAttributedPrompt(prompt_name="ExamplePrompt",
                                   args=args,
                                   outputs={})

# Run the prompt generation
df = dataset.run()

# Display the generated DataFrame
print(df)
```

## API Reference

### `DataFromAttributedPrompt`

```python
def __init__(self, prompt_name: str, args: dict, outputs: dict):
```
Initializes the `DataFromAttributedPrompt` class.

- `prompt_name` (str): The name of the prompt.
- `args` (dict): Arguments containing the LLM, instructions, and attributes.
- `outputs` (dict): Specifies how to handle the generated outputs.

```python
def register_input(self, name: str, help: str):
```
Registers an input for the generator.

- `name` (str): The name of the input.
- `help` (str): A description of the input.

```python
def register_output(self, name: str, help: str):
```
Registers an output for the generator.

- `name` (str): The name of the output.
- `help` (str): A description of the output.

```python
def prepare_prompts(self) -> List[str]:
```
Prepares multiple prompts based on the attributes.

Returns: 
- A list of formatted prompts generated from attribute combinations.

```python
def run(self) -> pd.DataFrame:
```
Generates synthetic data based on the attribute setup and returns it as a pandas DataFrame.

Returns: 
- A `pandas.DataFrame` containing the generated data.

```python
def save_to_excel(self, file_path: str, df: pd.DataFrame) -> None:
```
Saves the generated DataFrame to an Excel file.

- `file_path` (str): The path where the Excel file will be saved.
- `df` (pd.DataFrame): The DataFrame to be saved.
- Raises: `ValueError` if the DataFrame is empty or cannot be saved.
## Examples

### Generating Data Based on Attributes
```python
from indoxGen.synthCore import DataFromAttributedPrompt
from indoxGen.llms import IndoxApi
import os

# Load API key
os.environ["INDOX_API_KEY"] = "your_api_key"
LLM = IndoxApi(api_key=os.getenv("INDOX_API_KEY"))

# Define attributes
args = {
    "instruction": "Write a {tone} email about {topic}.",
    "attributes": {
        "tone": ["formal", "casual"],
        "topic": ["meeting", "project update"]
    },
    "llm": LLM
}

# Create an instance of DataFromAttributedPrompt
data_generator = DataFromAttributedPrompt(prompt_name="EmailPrompt",
                                          args=args,
                                          outputs={})

# Generate data
df = data_generator.run()

# Save to Excel
data_generator.save_to_excel("generated_emails.xlsx", df)
```

## Contributing
Contributions to improve `DataFromAttributedPrompt` are welcome. To contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature.
3. Add your changes and write tests if applicable.
4. Submit a pull request with a clear description of your changes.
