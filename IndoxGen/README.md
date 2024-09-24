# Synthetic Data Generator


## Overview

The Synthetic Data Generator is an advanced tool designed to create realistic, diverse, and high-quality synthetic data for various applications. It leverages the power of Large Language Models (LLMs) to generate data that closely mimics real-world datasets while maintaining privacy and confidentiality.

## Features

- **LLM-powered Generation**: Utilizes state-of-the-art language models to create diverse and realistic data points.
- **Dual Generator Types**:
  - **Judge-LLM Generator**: Employs an additional LLM to evaluate and ensure the quality of generated data.
  - **Standard Generator**: A lightweight version without the LLM judge for faster generation.
- **Customizable Output**: Generate data based on specific columns, example data, and user instructions.
- **Diversity Control**: Implements sophisticated mechanisms to ensure variety in the generated dataset.
- **Robust Error Handling**: Gracefully manages parsing errors and invalid outputs from LLMs.

## Installation

```bash
pip install synthetic-data-generator
```

## Quick Start

```python
from SynthCore import SyntheticDataGenerator

# Initialize the generator
generator = SyntheticDataGenerator(
    generator_llm=your_generator_llm,
    judge_llm=your_judge_llm,
    columns=['col1', 'col2', 'col3'],
    example_data=[{'col1': 'example1', 'col2': 'example2', 'col3': 'example3'}],
    user_instruction="Generate diverse medical record data"
)

# Generate data
generated_data = generator.generate_data(num_samples=100)

# Output to CSV
generated_data.to_csv('synthetic_data.csv', index=False)
```

## Configuration

The `SyntheticDataGenerator` class accepts the following parameters:

- `generator_llm`: The language model used for generating data.
- `judge_llm`: (Optional) The language model used for judging data quality.
- `columns`: List of column names for the synthetic data.
- `example_data`: List of example data points.
- `user_instruction`: Instruction for data generation.
- `real_data`: (Optional) List of real data points for reference.
- `diversity_threshold`: Threshold for determining data diversity (default: 0.7).
- `max_diversity_failures`: Maximum number of diversity failures before forcing acceptance (default: 20).
- `verbose`: Verbosity level (0 for minimal output, 1 for detailed feedback).

## Roadmap

- [x] Implement basic synthetic data generation
- [x] Add LLM-based judge for quality control
- [x] Improve diversity checking mechanism
- [ ] Integrate human feedback loop for continuous improvement
- [ ] Develop a web-based UI for easier interaction
- [ ] Support for more data types (images, time series, etc.)
- [ ] Implement differential privacy techniques
- [ ] Create plugin system for custom data generation rules
- [ ] Develop comprehensive documentation and tutorials

## Contributing

We welcome contributions! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) file for details on how to get started.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- Thanks to all contributors who have helped shape this project.
- Special thanks to the open-source AI community for their invaluable resources and tools.