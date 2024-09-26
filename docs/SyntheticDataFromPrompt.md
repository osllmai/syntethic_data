# SyntheticDataFromPrompt 

## Overview
`DataFromPrompt` is a Python class designed for generating structured data by leveraging a language model (LLM) based on a user-defined prompt and existing data (if provided). It can create new synthetic datasets or augment existing ones using the power of LLMs and then export the results to various formats like Excel.

## Table of Contents
- [Installation](#installation)
- [Language Model Setup](#language-model-setup)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Installation
To use the `DataFromPrompt` class, ensure that Python 3.9+ is installed. You can install the required dependencies by setting up your environment as follows:

```bash
pip install pandas loguru openai
```

Additionally, install the `indoxGen` package for interacting with the Indox language models:

```bash
pip install indoxGen
```

## Language Model Setup
The `DataFromPrompt` class requires a language model (LLM) for data generation. In this example, we are using the `IndoxApi` from the `indoxGen` library. Here's how to set up the LLM for generating synthetic data:

```python
from indoxGen.llms import IndoxApi
import os
from dotenv import load_dotenv

load_dotenv()
INDOX_API_KEY = os.getenv("INDOX_API_KEY")
LLM = IndoxApi(api_key=INDOX_API_KEY)
```

Make sure you have an API key from Indox or a compatible LLM provider to use this.

## Usage
The `DataFromPrompt` class allows you to generate a dataset based on a prompt. The dataset can either be brand new or augment an existing DataFrame.

Here's a simple usage example:

```python
from indoxGen.synthCore import DataFromPrompt
from indoxGen.llms import IndoxApi
import pandas as pd

LLM = IndoxApi(api_key=INDOX_API_KEY)
user_prompt = "Generate a dataset with 3 columns and 3 rows about astronomy."
instruction = DataGenerationPrompt.get_instruction(user_prompt)

# Initialize the DataFromPrompt class
data_generator = DataFromPrompt(
    prompt_name="Astronomy Dataset",
    args={
        "llm": LLM,
        "n": 1,
        "instruction": instruction
    },
    outputs={"generations": "generate"}
)

# Generate the dataset
generated_df = data_generator.run()
print(generated_df)

# Save the dataset to an Excel file
data_generator.save_to_excel("output_data.xlsx")
```

## API Reference

### `DataFromPrompt`
```python
__init__(self, prompt_name: str, args: dict, outputs: dict, dataframe: pd.DataFrame = None)
```
Initializes the `DataFromPrompt` class.

#### Parameters:
- `prompt_name` (str): The name of the prompt for generating data.
- `args` (dict): Contains arguments, including the LLM and other generation parameters.
    - `llm`: The language model instance used for data generation.
    - `n`: (optional) Number of generations to produce.
    - `instruction`: User instruction or prompt for the LLM.
- `outputs` (dict): Specifies the format or types of expected output.
- `dataframe` (pd.DataFrame): (optional) A DataFrame that can be augmented with new data.

### `run(self) -> pd.DataFrame`
Generates new data based on the provided prompt and returns the results as a Pandas DataFrame.

#### Returns:
- `pd.DataFrame`: A DataFrame containing the generated data.

#### Raises:
- `ValueError`: If the LLM's JSON response is not valid.

### `save_to_excel(self, file_path: str) -> None`
Saves the generated DataFrame to an Excel file.

#### Parameters:
- `file_path` (str): Path where the Excel file will be saved.

#### Raises:
- `ValueError`: If the DataFrame is empty or there is an issue saving the file.

## Examples

### Generating Data Based on an Astronomy Prompt
```python
from indoxGen.synthCore import DataFromPrompt
from indoxGen.llms import IndoxApi
import os
from dotenv import load_dotenv

# Load API key and initialize LLM
load_dotenv()
INDOX_API_KEY = os.getenv("INDOX_API_KEY")
LLM = IndoxApi(api_key=INDOX_API_KEY)

# Define the user prompt
user_prompt = "Generate a dataset with 3 columns and 3 rows about astronomy."
instruction = DataGenerationPrompt.get_instruction(user_prompt)

# Generate the data
data_generator = DataFromPrompt(
    prompt_name="Astronomy Dataset",
    args={"llm": LLM, "n": 1, "instruction": instruction},
    outputs={"generations": "generate"}
)

# Run the data generation and print the output
generated_df = data_generator.run()
print(generated_df)

# Save the data to an Excel file
data_generator.save_to_excel("astronomy_dataset.xlsx")
```

## Contributing
Contributions are welcome! To contribute, please:

1. Fork the repository.
2. Create a new branch for your feature.
3. Add your changes and write tests, if applicable.
4. Submit a pull request with a clear description of the changes.

