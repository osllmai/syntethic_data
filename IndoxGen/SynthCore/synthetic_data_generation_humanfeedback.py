import json
from typing import List, Dict, Any, Optional, Tuple
import random
from collections import defaultdict
import numpy as np
import pandas as pd
import re

class SyntheticDataGeneratorFeedback:
    def __init__(
            self,
            generator_llm: Any,
            judge_llm: Any,
            columns: List[str],
            example_data: List[Dict[str, Any]],
            user_instruction: str,
            real_data: Optional[List[Dict[str, Any]]] = None,
            diversity_threshold: float = 0.7,
            max_diversity_failures: int = 20,
            verbose: int = 0,
            feedback_range: Tuple[float, float] = (0.3, 0.6)
    ):
        """
        Initialize the SyntheticDataGeneratorFeedback.

        Args:
            generator_llm (Any): Language model for generating data.
            judge_llm (Any): Language model for judging data quality.
            columns (List[str]): List of column names for the synthetic data.
            example_data (List[Dict[str, Any]]): List of example data points.
            user_instruction (str): Instruction for data generation.
            real_data (Optional[List[Dict[str, Any]]]): Optional list of real data points.
            diversity_threshold (float): Threshold for determining data diversity.
            max_diversity_failures (int): Maximum number of diversity failures before forcing acceptance.
            verbose (int): Verbosity level (0 for minimal output, 1 for detailed feedback).
            feedback_range (Tuple[float, float]): Score range for feedback.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

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
        self.diversity_threshold = diversity_threshold
        self.max_diversity_failures = max_diversity_failures
        self.diversity_failure_count = 0
        self.verbose = verbose
        self.diversity_check_window = 5
        self.pending_review = []
        self.feedback_range = feedback_range

    def generate_data(self, num_samples: int) -> pd.DataFrame:
        """
        Generate synthetic data points.

        Args:
            num_samples: Number of data points to generate.

        Returns:
            DataFrame containing the generated data.
        """
        attempts = 0
        max_attempts = num_samples * 10

        while len(self.generated_data) +len(self.pending_review) < num_samples and attempts < max_attempts:
            attempts += 1
            generated = self._generate_single_data_point()
            if not generated:
                continue

            score = self._judge_data_point(generated)

            if score >= self.feedback_range[1] and self._is_diverse(generated):
                self.generated_data.append(generated)
                self.diversity_failure_count = 0
                if self.verbose >= 1:
                    print(f"Generated data point: {generated}")
            elif score >= self.feedback_range[1]:
                self._handle_diversity_failure(generated)

            elif self.feedback_range[0] < score < self.feedback_range[1] and self._is_diverse(generated):
                self.pending_review.append(generated)

            else:
                self._inform_generator(generated, score, "Low score")

            if self.verbose >= 1 and attempts % 10 == 0:
                print(f"Progress: {len(self.generated_data)}/{num_samples} data points generated. Attempts: {attempts}")

        if len(self.generated_data) < num_samples:
            print(
                f"Warning: Only generated {len(self.generated_data)} out of {num_samples} requested samples after"
                f" {attempts} attempts. Use 'user_review_and_regenerate' method for review and accept or regenerate rejected data with range score: {self.feedback_range} ")

        self.diversity_threshold = 0.7

        return self._convert_to_dataframe()

    def user_review_and_regenerate(
            self,
            accepted_rows: List[int],
            regenerate_rows: List[int],
            regeneration_feedback: str,
            min_score: float
    ) -> pd.DataFrame:
        """
        Review and regenerate synthetic data based on feedback.

        Args:
            accepted_rows (List[int]): Indices of rows to accept.
            regenerate_rows (List[int]): Indices of rows to regenerate.
            regeneration_feedback (str): Feedback for regeneration.
            min_score (float): Minimum score for accepting regenerated data.

        Returns:
            pd.DataFrame: DataFrame containing accepted and regenerated data.
        """

        accepted = []
        to_regenerate = []

        # If no rows are specified, accept all pending rows by default
        if not accepted_rows and not regenerate_rows:
            accepted_rows = list(range(len(self.pending_review)))

        for i in sorted(accepted_rows + regenerate_rows, reverse=True):
            if i in accepted_rows:
                accepted.append(self.pending_review[i])
                self.pending_review.pop(i)
            elif i in regenerate_rows:
                to_regenerate.append(self.pending_review[i])

        self.generated_data.extend(accepted)

        for data in to_regenerate:
            regenerated = self._regenerate_data_point(data, regeneration_feedback)
            print(regenerated)
            new_score = self._judge_data_point(regenerated)

            if new_score >= min_score and self._is_diverse(regenerated):
                self.generated_data.append(regenerated)
                # Only remove from pending_review if it's added to generated_data
                self.pending_review = [d for d in self.pending_review if d != data]
            else:
                print(f"Regenerated data point still has a low score ({new_score}). Keeping in pending review.")

        return self._convert_to_dataframe()

    def _regenerate_data_point(self, original_data: Dict[str, Any], feedback: str) -> Dict[str, Any]:
        """
        Regenerate a single data point based on feedback.

        Args:
            original_data (Dict[str, Any]): Original data to regenerate.
            feedback (str): Feedback for regeneration.

        Returns:
            Dict[str, Any]: Regenerated data point.
        """

        system_prompt = "You are an advanced synthetic data generator. Regenerate the given data point based on the feedback provided."

        # Start building the prompt
        prompt = f"Original data: {json.dumps(original_data)}\n"
        prompt += f"User feedback: {feedback}\n"
        prompt += f"User instruction: {self.user_instruction}\n"
        prompt += "Regenerate this data point, addressing the feedback while maintaining the overall structure and adhering to the original instructions.\n\n"

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
        for attempt in range(3):
            try:
                generated = self.generator_llm.chat(prompt, system_prompt=system_prompt, temperature=1.3)
                # Find the first '{' and last '}' to extract the JSON object
                start = generated.find('{')
                end = generated.rfind('}')
                if start != -1 and end != -1 and start < end:
                    json_str = generated[start:end + 1]
                    # Replace single quotes with double quotes for property names
                    json_str = re.sub(r"(\w+):", r'"\1":', json_str)
                    # Remove any control characters
                    json_str = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', json_str)
                    # Ensure all string values are properly quoted
                    json_str = re.sub(r': *([^",\{\}\[\]]+)([,\}])', r': "\1"\2', json_str)
                    data = json.loads(json_str)

                    if set(self.columns).issubset(data.keys()):
                        return data
                    else:
                        missing_columns = set(self.columns) - set(data.keys())
                        if self.verbose >= 1:
                            print(f"Generated data is missing columns: {missing_columns}")
                else:
                    if self.verbose >= 1:
                        print(f"Failed to find valid JSON object in generated text (Attempt {attempt + 1}/3)")
            except json.JSONDecodeError as e:
                if self.verbose >= 1:
                    print(f"Failed to parse generated data (Attempt {attempt + 1}/3): {str(e)}")
                    print(f"Problematic JSON string: {json_str}")

            if self.verbose >= 1 and attempt < 2:
                print(f"Retrying generation (Attempt {attempt + 2}/3)...")

        if self.verbose >= 1:
            print("Max attempts reached. Skipping this data point.")
        return {}


    def _generate_single_data_point(self) -> Dict[str, Any]:
        """
        Generate a single data point using the generator LLM.

        Returns:
            Dict[str, Any]: A generated data point.
        """
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
        """
        Calculate statistics for each column in the dataset.

        Returns:
            Dict[str, Dict[str, Any]]: Column statistics.
        """
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
        """
        Create a prompt for generating synthetic data.

        This method generates a prompt for the LLM (Language Model) to produce diverse and realistic synthetic data points
        based on the provided columns, user instructions, statistical information, and example data.

        Returns:
            str: A formatted prompt string to be passed to the LLM for data generation.
        """
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
        """
        Check if a new data point is diverse compared to existing data.

        Args:
            new_data (Dict[str, Any]): The new data point to check.

        Returns:
            bool: Whether the data point is diverse.
        """
        from sklearn.metrics.pairwise import cosine_similarity

        total_existing_data = self.generated_data + self.pending_review
        if len(total_existing_data) < 2:
            return True

        new_text = json.dumps(new_data)
        existing_texts = [json.dumps(data) for data in total_existing_data[-self.diversity_check_window:]]

        all_texts = existing_texts + [new_text]

        tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

        mean_similarity = np.mean(cosine_similarities)
        std_similarity = np.std(cosine_similarities)

        if mean_similarity < self.diversity_threshold:
            return True
        elif mean_similarity < self.diversity_threshold + 0.1 and std_similarity > 0.1:
            return True
        else:
            return False

    def _handle_diversity_failure(self, generated: Dict[str, Any]) -> None:
        """
        Handle diversity failure when generating data.

        Args:
            generated (Dict[str, Any]): Generated data point that failed the diversity check.
        """
        self.diversity_failure_count += 1
        if self.verbose >= 1:
            print(f"Generated data is not diverse. Retrying... (Failure count: {self.diversity_failure_count})")
        if self.diversity_failure_count >= self.max_diversity_failures:
            if self.verbose >= 1:
                print("Max diversity failures reached. Forcing acceptance of this data point.")
            self.generated_data.append(generated)
            self.diversity_failure_count = 0
        elif self.diversity_failure_count % 5 == 0:
            # Every 5 failures, slightly increase the diversity threshold
            self.diversity_threshold += 0.05
            if self.verbose >= 1:
                print(f"Increased diversity threshold to {self.diversity_threshold}")

    def _judge_data_point(self, data: Dict[str, Any]) -> float:
        """
        Judge the quality of a generated data point.

        Args:
            data (Dict[str, Any]): Data point to judge.

        Returns:
            float: Quality score of the data point.
        """
        system_prompt = ("You are a data quality judge. Evaluate the given data based on the criteria and return a "
                         "score between 0 and 1. It's important to only send score without any description")
        criteria = self._create_judge_criteria()
        prompt = (f"Data to evaluate: {json.dumps(data)}\n\nCriteria:\n{criteria}\n\nProvide a numeric score between 0 "
                  f"and 1.")

        score_str = self.judge_llm.chat(prompt, system_prompt=system_prompt, temperature=0.2)
        try:
            return float(score_str)
        except ValueError:
            if self.verbose >= 1:
                print(f"Failed to parse judge score: {score_str}")
            return 0.5

    def _inform_generator(self, data: Dict[str, Any], score: float, reason: str) -> None:
        """
        Inform the generator about the quality of generated data.

        Args:
            data (Dict[str, Any]): The generated data.
            score (float): Quality score of the data.
            reason (str): Reason for feedback.
        """
        feedback = f"Generated data: {json.dumps(data)}\nScore: {score}\nReason: {reason}"
        self.feedback_history.append(feedback)
        if self.verbose >= 1:
            print(f"Feedback for generator: {feedback}")

    def _convert_to_dataframe(self) -> pd.DataFrame:
        """
        Convert generated data into a pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame of generated data.
        """
        return pd.DataFrame(self.generated_data)

    def _create_judge_criteria(self) -> str:
        """
        Create criteria for judging generated data.

        Returns:
            str: Criteria for judging data.
        """
        criteria = "Evaluate the generated data based on the following criteria:\n"
        criteria += f"1. Adheres to the user instruction: {self.user_instruction}\n"
        criteria += "2. Contains all required columns.\n"
        criteria += "3. Data types match the example data.\n"
        criteria += "4. Values are plausible and make sense within the context.\n"
        criteria += "5. Avoids clear personal information like full names, addresses.\n"
        criteria += "6. Demonstrates significant creativity while maintaining realism.\n"
        criteria += "7. Shows high diversity compared to previously generated data.\n"
        criteria += ("Return a score between 0 and 1, where 1 is perfect. Only return the numeric score without any "
                     "additional text.")
        return criteria

