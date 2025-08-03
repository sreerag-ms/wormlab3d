import sys
import argparse
from wormlab3d.midlines3d.args_finder.parse import parse_arguments, RuntimeArgs, SourceArgs, ParameterArgs
from wormlab3d.midlines3d.midline3d_finder import Midline3DFinder

def _load_args_file():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--argsfile', type=str, help='Path to txt file with args')
    args, _ = parser.parse_known_args()
    if not args.argsfile:
        return None

    lines = []
    with open(args.argsfile, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            lines.append(line)
    return lines


def _process_args_file(lines: list):
    proto = argparse.ArgumentParser(add_help=False)
    RuntimeArgs.add_args(proto)
    SourceArgs.add_args(proto)
    ParameterArgs.add_args(proto)

    flag_only = set(
        action.dest.replace('_', '-')
        for action in proto._actions
        if action.nargs == 0
    )

    new_argv = [sys.argv[0]]
    for entry in lines:
        if '=' in entry:
            key, val = entry.split('=', 1)
            key = key.lstrip('-')
            if key in flag_only:
                if val.lower() == 'true':
                    new_argv.append(f'--{key}')
            else:
                new_argv.append(entry)
        else:
            new_argv.append(entry)

    sys.argv = new_argv

def train():
    lines = _load_args_file()
    if lines is not None:
        _process_args_file(lines)

    runtime_args, source_args, parameter_args = parse_arguments()

    # Construct finder
    manager = Midline3DFinder(
        runtime_args=runtime_args,
        source_args=source_args,
        parameter_args=parameter_args
    )

    # Process the trial
    manager.process_trial()


if __name__ == '__main__':
    train()
