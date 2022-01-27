from wormlab3d.postures.args.parse import parse_arguments
from wormlab3d.postures.generate_midline3d_dataset import generate_midline3d_dataset


def generate_dataset():
    dataset_args = parse_arguments()
    generate_midline3d_dataset(dataset_args)


if __name__ == '__main__':
    generate_dataset()
