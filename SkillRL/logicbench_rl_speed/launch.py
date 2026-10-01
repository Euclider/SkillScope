"""Use the registered compute profile without rewriting the frozen study setting."""
import argparse
from pathlib import Path
import sys

from .runtime import apply_options, child_args, validate


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--setting', type=Path)
    args, _ = parser.parse_known_args()
    validate(args.root)
    if args.setting is None:
        sys.argv.extend(['--setting',str(args.root.resolve()/'setting.json')])
    from phase3 import logicbench_loop as loop
    original_config, original_command = loop.training_configuration, loop.command

    def configuration(*args, **kwargs):
        return apply_options(original_config(*args, **kwargs), start=kwargs['start'])

    def command(root, tag, args):
        return original_command(root, tag, child_args(args))

    loop.training_configuration = configuration
    loop.command = command
    loop.main()


if __name__ == '__main__':
    main()
