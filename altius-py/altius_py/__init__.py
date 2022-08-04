from .altius_py import load, session


class InferenceSession:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = load(model_path)
        self.session = session(self.model)

    def run(self, output, input):
        assert output is None
        return self.session.run(input)
