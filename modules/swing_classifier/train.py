import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from utils.load import load_yaml
from model import get_model


def parser():
    parser = argparse.ArgumentParser('Classification Argument')
    parser.add_argument('--configfile', type=str, default='config/config.yaml', help='config file')
    parser.add_argument('--eval', action='store_true', help='eval mode')
    args = parser.parse_args()
    return args

def run(args):
    """Builds model, loads data, trains and evaluates"""
    config = load_yaml(args.configfile)

    model = get_model(config['Swing_clssifier'])
    model.load_data(is_eval=args.eval)
    model.build(is_eval=args.eval)
    
    if args.eval:
        model.evaluate()
    else:
        model.train()

if __name__ == '__main__':
    args = parser()
    run(args)

