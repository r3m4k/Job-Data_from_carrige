import argparse

parser = argparse.ArgumentParser()


parser.add_argument("-fh", "--final_height", type=float, default=0)
print(parser.parse_args().final_height)


