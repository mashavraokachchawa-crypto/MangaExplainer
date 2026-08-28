"""Base abstractions for pipeline stages (skeleton)."""


class Stage:
    NAME = ""
    INPUT_DIR = ""
    OUTPUT_DIR = ""

    def __init__(self, name, input_dir, output_dir):
        self.NAME = name
        self.INPUT_DIR = input_dir
        self.OUTPUT_DIR = output_dir

    @property
    def name(self):
        return self.NAME

    @property
    def input_dir(self):
        return self.INPUT_DIR

    @property
    def output_dir(self):
        return self.OUTPUT_DIR

    def run(self, ctx):
        raise NotImplementedError(self.NAME)