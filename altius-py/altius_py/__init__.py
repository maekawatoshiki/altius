from .altius_py import load, session


class InferenceSession:
    """
    ``InferenceSession`` is the class used to run a model.
    """

    def __init__(self, model_path, enable_profile=False, intra_op_num_threads=1, backend="interpreter"):
        self.model_path = model_path
        self.model = load(model_path)
        self.session = session(self.model, enable_profile, intra_op_num_threads, backend)

    def run(self, output, input):
        """
        Compute the predictions.

        Args:
            output (Optional[list[str]]): Name of the outputs, but must be None for now.
            input (dict[str, numpy.ndarray]): Dictionary ``{ input_name: input_value }``.

        Returns:
            list[numpy.ndarray]: Output values.
        """

        assert output is None
        return self.session.run(input)
