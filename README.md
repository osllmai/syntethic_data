# IndoxGen: Comprehensive Synthetic Data Generation Framework

[![License](https://img.shields.io/github/license/osllmai/indoxGen)](https://github.com/osllmai/indoxGen/blob/main/LICENSE)
[![Python](https://img.shields.io/pypi/pyversions/indoxGen.svg)](https://pypi.org/project/indoxGen/)

[![Discord](https://img.shields.io/discord/1223867382460579961?label=Discord&logo=Discord&style=social)](https://discord.com/invite/ossllmai)
[![GitHub stars](https://img.shields.io/github/stars/osllmai/indoxGen?style=social)](https://github.com/osllmai/indoxGen)

<p align="center">
  <a href="https://osllm.ai">Official Website</a> &bull; <a href="https://docs.osllm.ai/index.html">Documentation</a> &bull; <a href="https://discord.gg/qrCc56ZR">Discord</a>
</p>

<p align="center">
  <b>NEW:</b> <a href="https://docs.google.com/forms/d/1CQXJvxLUqLBSXnjqQmRpOyZqD6nrKubLz2WTcIJ37fU/prefill">Subscribe to our mailing list</a> for updates and news!
</p>

## Overview

IndoxGen is a state-of-the-art, enterprise-ready framework designed for generating high-fidelity synthetic data. It consists of two main components:

1. **IndoxGen Core**: Leverages advanced AI technologies, including Large Language Models (LLMs) and human feedback loops, for flexible and precise synthetic data creation.
2. **IndoxGen-Tensor**: Utilizes Generative Adversarial Networks (GANs) powered by TensorFlow for generating complex tabular data.

Together, these components offer a comprehensive solution for synthetic data generation across various domains and use cases.

## Components

### 1. IndoxGen

[![PyPI](https://badge.fury.io/py/indoxGen.svg)](https://pypi.org/project/indoxGen/0.0.3/)
[![Downloads](https://static.pepy.tech/badge/indoxGen)](https://pepy.tech/project/indoxGen)

Key Features:
- Multiple generation pipelines (SyntheticDataGenerator, SyntheticDataGeneratorHF, DataFromPrompt)
- Human-in-the-loop feedback integration
- AI-driven diversity ensuring representative datasets
- Flexible I/O supporting various data sources and export formats
- Advanced learning techniques including few-shot learning

[Learn more about IndoxGen Core](#indoxgen-core)

### 2. IndoxGen-Tensor

[![PyPI](https://badge.fury.io/py/indoxGen-tensor.svg)](https://pypi.org/project/indoxGen-tensor/)
[![Downloads](https://static.pepy.tech/badge/indoxGen-tensor)](https://pepy.tech/project/indoxGen-tensor)

Key Features:
- GAN-based generation for high-fidelity synthetic data
- TensorFlow integration for efficient, GPU-accelerated training
- Flexible data handling supporting categorical, mixed, and integer columns
- Customizable GAN architecture
- Scalable generation for large volumes of synthetic data

[Learn more about IndoxGen-Tensor](#indoxgen-tensor)

## Installation

To install both components:

```bash
pip install indoxgen indoxgen-tensor
```

Or install them separately:

```bash
pip install indoxgen
pip install indoxgen-tensor
```

## Quick Start Guide

### IndoxGen

```python
from indoxGen.synthCore import SyntheticDataGenerator
from indoxGen.llms import OpenAi

columns = ["name", "age", "occupation"]
example_data = [
    {"name": "Alice Johnson", "age": 35, "occupation": "Manager"},
    {"name": "Bob Williams", "age": 42, "occupation": "Accountant"}
]

openai = OpenAi(api_key=OPENAI_API_KEY, model="gpt-4o-mini")
nemotron = OpenAi(api_key=NVIDIA_API_KEY, model="nvidia/nemotron-4-340b-instruct",
                  base_url="https://integrate.api.nvidia.com/v1")

generator = SyntheticDataGenerator(
    generator_llm=nemotron,
    judge_llm=openai,
    columns=columns,
    example_data=example_data,
    user_instruction="Generate diverse, realistic data including name, age, and occupation. Ensure variability in demographics and professions.",
    verbose=1
)

generated_data = generator.generate_data(num_samples=100)
```

### IndoxGen-Tensor

```python
from indoxGen_tensor import TabularGANConfig, TabularGANTrainer
import pandas as pd

data = pd.read_csv("data/Adult.csv")

categorical_columns = ["workclass", "education", "marital-status", "occupation",
                       "relationship", "race", "gender", "native-country", "income"]
mixed_columns = {"capital-gain": "positive", "capital-loss": "positive"}
integer_columns = ["age", "fnlwgt", "hours-per-week", "capital-gain", "capital-loss"]

config = TabularGANConfig(
    input_dim=200,
    generator_layers=[128, 256, 512],
    discriminator_layers=[512, 256, 128],
    learning_rate=2e-4,
    beta_1=0.5,
    beta_2=0.9,
    batch_size=128,
    epochs=50,
    n_critic=5
)

trainer = TabularGANTrainer(
    config=config,
    categorical_columns=categorical_columns,
    mixed_columns=mixed_columns,
    integer_columns=integer_columns
)

history = trainer.train(data, patience=15)
synthetic_data = trainer.generate_samples(50000)
```


## Use Cases

- Data Augmentation for Machine Learning
- Privacy-Preserving Data Sharing
- Software Testing and Quality Assurance
- Scenario Planning and Simulation
- Balancing Imbalanced Datasets

## Roadmap

- [x] Integrate IndoxGen and IndoxGen-Tensor for seamless workflow
- [ ] Develop a unified web-based UI for both components
- [ ] Implement advanced privacy-preserving techniques across both modules
- [ ] Extend support to more data types (images, time series, etc.)
- [ ] Create a plugin system for custom data generation rules
- [ ] Develop comprehensive documentation and tutorials covering both components

## Contributing

We welcome contributions to both IndoxGen Core and IndoxGen-Tensor! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) file for details on how to get started.

## License

Both IndoxGen Core and IndoxGen-Tensor are released under the MIT License. See [LICENSE.md](LICENSE.md) for more details.

---

IndoxGen - Empowering Data-Driven Innovation with Comprehensive Synthetic Data Generation