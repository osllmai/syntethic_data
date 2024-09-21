from typing import Optional, List, Dict, Any
import json


class SyntheticDataGenerator:
    def __init__(
            self,
            generator_llm,
            judge_llm,
            columns: List[str],
            example_data: List[Dict[str, Any]],
            real_data: Optional[List[Dict[str, Any]]] = None
    ):
        self.generator_llm = generator_llm
        self.judge_llm = judge_llm
        self.columns = columns
        self.example_data = example_data
        self.real_data = real_data
        self.generated_data = []
        self.feedback_history = []

    def generate_data(self, num_samples: int) -> List[Dict[str, Any]]:
        for _ in range(num_samples):
            while True:
                generated = self._generate_single_data_point()
                score = self._judge_data_point(generated)

                if score >= 0.9:  # Perfect score
                    self.generated_data.append(generated)
                    break
                elif score >= 0.5:  # Medium score
                    if self._ask_human_feedback(generated):
                        self.generated_data.append(generated)
                        break
                    else:
                        self._inform_generator(generated, score, "Human rejected")
                else:  # Low score
                    self._inform_generator(generated, score, "Low score")

        return self.generated_data

    def _generate_single_data_point(self) -> Dict[str, Any]:
        system_prompt = "You are a synthetic data generator. Generate realistic data based on the given examples and criteria."
        prompt = self._create_generation_prompt()
        generated = self.generator_llm.chat(prompt, system_prompt=system_prompt, temperature=0.7)
        try:
            return json.loads(generated)
        except json.JSONDecodeError:
            print(f"Failed to parse generated data: {generated}")
            return {}

    def _judge_data_point(self, data: Dict[str, Any]) -> float:
        system_prompt = "You are a data quality judge. Evaluate the given data based on the criteria and return a score between 0 and 1."
        criteria = self._create_judge_criteria()
        prompt = f"Data to evaluate: {json.dumps(data)}\n\nCriteria:\n{criteria}\n\nProvide only a numeric score between 0 and 1."
        score_str = self.judge_llm.chat(prompt, system_prompt=system_prompt, temperature=0.2)
        try:
            return float(score_str)
        except ValueError:
            print(f"Failed to parse judge score: {score_str}")
            return 0.0

    def _ask_human_feedback(self, data: Dict[str, Any]) -> bool:
        print("\nPlease review this generated data point:")
        for col, value in data.items():
            print(f"{col}: {value}")
        return input("Is this data acceptable? (y/n): ").lower() == 'y'

    def _inform_generator(self, data: Dict[str, Any], score: float, reason: str):
        feedback = f"Generated data: {json.dumps(data)}\nScore: {score}\nReason: {reason}"
        self.feedback_history.append(feedback)
        print(f"Feedback for generator: {feedback}")

    def _create_generation_prompt(self) -> str:
        prompt = f"Generate synthetic data with the following columns: {', '.join(self.columns)}\n"
        prompt += "The data should be similar to the following examples:\n\n"
        for example in self.example_data:
            prompt += json.dumps(example) + "\n"
        if self.real_data:
            prompt += "\nAdditional real data for reference:\n"
            for real in self.real_data:
                prompt += json.dumps(real) + "\n"
        if self.feedback_history:
            prompt += "\nPrevious feedback:\n"
            prompt += "\n".join(self.feedback_history[-3:])  # Include last 3 feedback items
        prompt += "\nGenerate a single data point as a JSON object."
        return prompt

    def _create_judge_criteria(self) -> str:
        criteria = "Evaluate the generated data based on the following criteria:\n"
        criteria += "1. Contains all required columns\n"
        criteria += "2. Data types match the example data\n"
        criteria += "3. Values are plausible and coherent\n"
        criteria += "4. Absence of personally identifiable information\n"
        criteria += "5. Similarity to the example and real data patterns\n"
        criteria += "Return a score between 0 and 1, where 1 is perfect."
        return criteria


# Example usage with mock LLMs
# class MockLLM(LLM):
#     def chat(self, prompt: str, system_prompt: str = "You are a helpful assistant",
#              max_tokens: Optional[int] = None, temperature: float = 0.2,
#              frequency_penalty: Optional[float] = None, presence_penalty: Optional[float] = None,
#              top_p: Optional[float] = None, stream: Optional[bool] = None) -> str:
#         if "Generate synthetic data" in prompt:
#             return json.dumps({"name": "John Doe", "age": 30, "occupation": "Engineer"})
#         elif "Evaluate the generated data" in prompt:
#             return "0.85"
#         else:
#             return "I don't understand the prompt."


# Set up the generator
columns = ["name", "age", "occupation"]
example_data = [
    {"name": "Alice Johnson", "age": 35, "occupation": "Manager"},
    {"name": "Bob Williams", "age": 42, "occupation": "Accountant"}
]

generator = SyntheticDataGenerator(
    generator_llm=MockLLM(),
    judge_llm=MockLLM(),
    columns=columns,
    example_data=example_data
)

# Generate data
generated_data = generator.generate_data(num_samples=3)

# Print final generated data
print("\nFinal generated data:")
for data in generator.generated_data:
    print(json.dumps(data, indent=2))
