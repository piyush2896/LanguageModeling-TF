import argparse

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_txt', type=str, default='The moon would be black tonight',
                        help='Starting Text Default - "The moon would be black tonight"')
    parser.add_argument('--seq_len', type=int, default=100,
                        help="Number of words in the output sequence, Default - 100")
    return parser.parse_args()
