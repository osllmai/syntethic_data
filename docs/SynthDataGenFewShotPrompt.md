
# FewShotPrompt

## Overview
**FewShotPrompt** is a Python class designed for generating synthetic data based on few-shot learning examples and user-provided instructions. It utilizes language models to generate diverse datasets, leveraging pre-existing examples for guidance. The class supports outputting the generated data as a pandas DataFrame and allows saving the results to an Excel file.

## Table of Contents
- [Installation](#installation)
- [Language Model Setup](#language-model-setup)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Contributing](#contributing)

## Installation
To use **FewShotPrompt**, you need to have Python 3.9+ installed. You can install the required package using pip:

```bash
pip install indoxGen
```

## Language Model Setup
**FewShotPrompt** requires a language model for generating synthetic data. The `indoxGen` library provides a unified interface for various language models. Here's how to set up a language model for use in the class:

```python
from indoxGen.llms import IndoxApi

# Setup for IndoxApi model
llm = IndoxApi(api_key=INDOX_API_KEY)
```

The **indoxGen** library supports various models, including:
- OpenAI
- Mistral
- Ollama
- Google AI
- Hugging Face models

Additionally, **indoxGen** provides a router for OpenAI, allowing for easy switching between different models.

## Usage
Here's a basic example of how to use **FewShotPrompt**:

```python
from indoxGen.synthCore import FewShotPrompt
from indoxGen.llms import IndoxApi

# Setup language model
llm = IndoxApi(api_key=INDOX_API_KEY)

# Few-shot examples
examples = [
    {
        "input": "Generate a dataset with 3 columns and 2 rows about biology.",
        "output": '[{"Species": "Human", "Cell Count": 37.2, "Age": 30}, {"Species": "Mouse", "Cell Count": 3.2, "Age": 2}]'
    },
    {
        "input": "Generate a dataset with 3 columns and 2 rows about chemistry.",
        "output": '[{"Element": "Hydrogen", "Atomic Number": 1, "Weight": 1.008}, {"Element": "Oxygen", "Atomic Number": 8, "Weight": 15.999}]'
    }
]

user_prompt = "Generate a dataset with 3 columns and 2 rows about astronomy."

# Create FewShotPrompt instance
data_generator = FewShotPrompt(
    prompt_name="Generate Astronomy Dataset",
    args={
        "llm": llm,
        "n": 1,
        "instruction": user_prompt,
    },
    outputs={"generations": "generate"},
    examples=examples
)

# Generate and save the dataset
generated_df = data_generator.run()
print(generated_df)
data_generator.save_to_excel("output_data.xlsx", generated_df)
```

## API Reference

### FewShotPrompt

```python
def __init__(self, prompt_name: str, args: dict, outputs: dict, examples: List[Dict[str, str]]):
```
Initializes the **FewShotPrompt** class.

- `prompt_name` (str): The name of the prompt.
- `args` (dict): Arguments containing the LLM and instructions.
- `outputs` (dict): Specifies how to handle the generated outputs.
- `examples` (List[Dict[str, str]]): A list of few-shot examples (input-output pairs).

---

```python
def prepare_prompt(self) -> str:
```
Prepares the full prompt by including the examples and user instructions.

- Returns: A string representing the prompt to be sent to the LLM.

---

```python
def run(self) -> pd.DataFrame:
```
Generates synthetic data based on the few-shot setup and returns it as a pandas DataFrame.

- Returns: A pandas DataFrame containing the generated data.

---

```python
def save_to_excel(self, file_path: str, df: pd.DataFrame) -> None:
```
Saves the generated DataFrame to an Excel file.

- `file_path` (str): The path where the Excel file will be saved.
- `df` (pd.DataFrame): The DataFrame to be saved.
- Raises: `ValueError` if the DataFrame is empty or cannot be saved.

## Examples

### Generating Astronomy Dataset
```python
from indoxGen.synthCore import FewShotPrompt
from indoxGen.llms import IndoxApi

# Setup language model
llm = IndoxApi(api_key=INDOX_API_KEY)

# Few-shot examples
examples = [
    {
        "input": "Generate a dataset with 3 columns and 2 rows about biology.",
        "output": '[{"Species": "Human", "Cell Count": 37.2, "Age": 30}, {"Species": "Mouse", "Cell Count": 3.2, "Age": 2}]'
    },
    {
        "input": "Generate a dataset with 3 columns and 2 rows about chemistry.",
        "output": '[{"Element": "Hydrogen", "Atomic Number": 1, "Weight": 1.008}, {"Element": "Oxygen", "Atomic Number": 8, "Weight": 15.999}]'
    }
]

user_prompt = "Generate a dataset with 3 columns and 2 rows about astronomy."

# Create FewShotPrompt instance
data_generator = FewShotPrompt(
    prompt_name="Generate Astronomy Dataset",
    args={
        "llm": llm,
        "n": 1,
        "instruction": user_prompt,
    },
    outputs={"generations": "generate"},
    examples=examples
)

# Generate and save the dataset
generated_df = data_generator.run()
print(generated_df)
data_generator.save_to_excel("output_data.xlsx", generated_df)
```

## Contributing
Contributions to improve **FewShotPrompt** are welcome. Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature.
3. Add your changes and write tests if applicable.
4. Submit a pull request with a clear description of your changes.
