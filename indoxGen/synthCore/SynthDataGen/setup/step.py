class Step:
    """
    Base class for steps in the synthetic data generation pipeline.
    Attributes:
        inputs (Dict[str, str]): Registered inputs for the step.
        outputs (Dict[str, str]): Registered outputs for the step.
    """

    def __init__(self):
        """Initializes the Step with empty inputs and outputs."""
        self.inputs = {}
        self.outputs = {}

    def register_input(self, name: str, help: str):
        """
        Registers an input for the step.

        Args:
            name (str): The name of the input.
            help (str): Description or help text for the input.
        """
        self.inputs[name] = help

    def register_output(self, name: str, help: str):
        """
        Registers an output for the step.

        Args:
            name (str): The name of the output.
            help (str): Description or help text for the output.
        """
        self.outputs[name] = help

    def run(self):
        """
        Abstract method to be implemented by subclasses.

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """
        raise NotImplementedError("Subclasses should implement this method.")

