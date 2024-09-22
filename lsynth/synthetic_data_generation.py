import json
from typing import List, Dict, Any, Optional
import random
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


class SyntheticDataGenerator:
    def __init__(
            self,
            generator_llm,
            judge_llm,
            columns: List[str],
            example_data: List[Dict[str, Any]],
            user_instruction: str,
            real_data: Optional[List[Dict[str, Any]]] = None
    ):
        self.generator_llm = generator_llm
        self.judge_llm = judge_llm
        self.columns = columns
        self.example_data = example_data
        self.user_instruction = user_instruction
        self.real_data = real_data or []
        self.generated_data = []
        self.feedback_history = []
        self.column_stats = self._calculate_column_stats()
        self.vectorizer = TfidfVectorizer()
        self.diversity_threshold = 0.4
        self.diversity_failure_count = 0
        self.max_diversity_failures = 10

    def generate_data(self, num_samples: int) -> pd.DataFrame:
        attempts = 0
        max_attempts = num_samples * 5

        while len(self.generated_data) < num_samples and attempts < max_attempts:
            attempts += 1
            generated = self._generate_single_data_point()
            if not generated:
                continue

            score = self._judge_data_point(generated)

            if score >= 0.6:
                if self._is_diverse(generated):
                    self.generated_data.append(generated)
                    self.diversity_failure_count = 0
                    print(f"Generated diverse data point: {generated}")
                else:
                    self.diversity_failure_count += 1
                    print(f"Generated data is not diverse. Retrying... (Failure count: {self.diversity_failure_count})")
                    if self.diversity_failure_count >= self.max_diversity_failures:
                        print("Max diversity failures reached. Forcing acceptance of this data point.")
                        self.generated_data.append(generated)
                        self.diversity_failure_count = 0
            else:
                self._inform_generator(generated, score, "Low score")

            if attempts % 10 == 0:
                print(f"Progress: {len(self.generated_data)}/{num_samples} data points generated. Attempts: {attempts}")

        if len(self.generated_data) < num_samples:
            print(
                f"Warning: Only generated {len(self.generated_data)} out of {num_samples} requested samples after {attempts} attempts.")

        return self._convert_to_dataframe()

    def _generate_single_data_point(self) -> Dict[str, Any]:
        system_prompt = "You are an advanced synthetic data generator. Create diverse and realistic data based on the given examples, criteria, and user instruction. Your response must be a valid JSON object."
        prompt = self._create_generation_prompt()

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                generated = self.generator_llm.chat(prompt, system_prompt=system_prompt, temperature=1.3)
                json_start = generated.find('{')
                json_end = generated.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    json_str = generated[json_start:json_end]
                    data = json.loads(json_str)

                    if all(col in data for col in self.columns):
                        return data
                    else:
                        missing_columns = set(self.columns) - set(data.keys())
                        print(f"Generated data is missing columns: {missing_columns}")
                else:
                    print("No valid JSON object found in the generated data")
            except json.JSONDecodeError as e:
                print(f"Failed to parse generated data (Attempt {attempt + 1}/{max_attempts}): {str(e)}")

            if attempt < max_attempts - 1:
                print(f"Retrying generation (Attempt {attempt + 2}/{max_attempts})...")

        print("Max attempts reached. Skipping this data point.")
        return {}

    def _calculate_column_stats(self) -> Dict[str, Dict[str, Any]]:
        stats = defaultdict(lambda: {'min': float('inf'), 'max': float('-inf'), 'mean': 0, 'unique_values': set()})
        all_data = self.example_data + self.real_data

        for data in all_data:
            for col, value in data.items():
                if isinstance(value, (int, float)):
                    stats[col]['min'] = min(stats[col]['min'], value)
                    stats[col]['max'] = max(stats[col]['max'], value)
                    stats[col]['mean'] += value
                elif isinstance(value, str):
                    stats[col]['unique_values'].add(value)

        for col in stats:
            if 'mean' in stats[col]:
                stats[col]['mean'] /= len(all_data)
                stats[col]['std'] = np.std([data[col] for data in all_data if isinstance(data.get(col), (int, float))])

        return dict(stats)

    def _create_generation_prompt(self) -> str:
        prompt = f"Generate diverse synthetic data with the following columns: {', '.join(self.columns)}.\n"
        prompt += f"User instruction: {self.user_instruction}\n"
        prompt += "Ensure that each generated data point is unique and significantly different from the previous ones.\n"
        prompt += "The data should be realistic and inspired by the given examples, but with substantial variations.\n\n"

        prompt += "Statistical information for numerical columns (use as a guide, not strict rules):\n"
        for col, stats in self.column_stats.items():
            if 'mean' in stats:
                prompt += f"{col}: min={stats['min']}, max={stats['max']}, mean={stats['mean']:.2f}, std={stats['std']:.2f}\n"

        prompt += "\nExample values for categorical columns:\n"
        for col, stats in self.column_stats.items():
            if 'unique_values' in stats:
                prompt += f"{col}: {', '.join(list(stats['unique_values'])[:10])}\n"

        shuffled_examples = random.sample(self.example_data + self.real_data,
                                          min(5, len(self.example_data) + len(self.real_data)))
        prompt += "\nExample data points:\n"
        for example in shuffled_examples:
            prompt += json.dumps(example) + "\n"

        if self.generated_data:
            prompt += "\nRecently generated data (generate something significantly different):\n"
            for data in self.generated_data[-3:]:
                prompt += json.dumps(data) + "\n"

        prompt += "\nGenerate a single, unique data point as a JSON object. Be creative and ensure high diversity while staying realistic."
        return prompt

    def _is_diverse(self, new_data: Dict[str, Any]) -> bool:
        if len(self.generated_data) < 2:
            return True

        new_text = json.dumps(new_data)
        existing_texts = [json.dumps(data) for data in self.generated_data[-10:]]  # Compare with last 10 data points

        all_texts = existing_texts + [new_text]
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

        return np.mean(cosine_similarities) < self.diversity_threshold  # Changed to mean similarity

    def _judge_data_point(self, data: Dict[str, Any]) -> float:
        system_prompt = "You are a data quality judge. Evaluate the given data based on the criteria and return a score between 0 and 1."
        criteria = self._create_judge_criteria()
        prompt = f"Data to evaluate: {json.dumps(data)}\n\nCriteria:\n{criteria}\n\nProvide a numeric score between 0 and 1."

        score_str = self.judge_llm.chat(prompt, system_prompt=system_prompt, temperature=0.2)
        try:
            score = float(score_str)
            return score
        except ValueError:
            print(f"Failed to parse judge score: {score_str}")
            return 0.5

    def _inform_generator(self, data: Dict[str, Any], score: float, reason: str):
        feedback = f"Generated data: {json.dumps(data)}\nScore: {score}\nReason: {reason}"
        self.feedback_history.append(feedback)
        print(f"Feedback for generator: {feedback}")

    def _convert_to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.generated_data)

    def _create_judge_criteria(self) -> str:
        criteria = "Evaluate the generated data based on the following criteria:\n"
        criteria += f"1. Adheres to the user instruction: {self.user_instruction}\n"
        criteria += "2. Contains all required columns.\n"
        criteria += "3. Data types match the example data.\n"
        criteria += "4. Values are plausible and make sense within the context.\n"
        criteria += "5. Avoids clear personal information like full names, addresses.\n"
        criteria += "6. Demonstrates significant creativity while maintaining realism.\n"
        criteria += "7. Shows high diversity compared to previously generated data.\n"
        criteria += "Return a score between 0 and 1, where 1 is perfect. Only return the numeric score without any additional text."
        return criteria
