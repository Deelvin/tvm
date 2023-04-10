import bert_base_uncased

from bert_utils import Profiler, get_encoded_input, NUM_ITERATIONS


def main():
    model = bert_base_uncased.OctomizedModel()
    # model.benchmark()

    encoded_input = get_encoded_input(return_tensors="np").values()
    with Profiler("Octomizer", show_latency=True, iterations_number=NUM_ITERATIONS):
        output = model.run(*encoded_input)


if __name__ == '__main__':
    main()
