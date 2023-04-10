import time

from transformers import BertModel, BertTokenizer


NUM_ITERATIONS = 10


class Profiler:
    def __init__(self, name="", show_latency=False, iterations_number=1):
        self.name = name
        self.show_latency = show_latency
        self.iterations_number = iterations_number

    def __enter__(self):
        self.start_time = time.perf_counter()

    def __exit__(self, type, value, traceback):
        end_time = time.perf_counter()
        if self.show_latency:
            print(
                "[{}] elapsed time: {:.3f} ms ({} iterations)".format(
                    self.name,
                    (end_time - self.start_time) * 1000 / self.iterations_number,
                    self.iterations_number,
                    )
            )


def get_pytorch_model():
    model = BertModel.from_pretrained("bert-base-uncased", return_dict=False)
    model = model.eval()
    return model


def get_encoded_input(return_tensors):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    text = "Replace me by any text you'd like. " * 12
    encoded_input = tokenizer(text, return_tensors=return_tensors)
    return encoded_input
