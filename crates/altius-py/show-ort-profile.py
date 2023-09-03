import json
import argparse
from collections import defaultdict

from tabulate import tabulate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", help="onnxruntime profile json", type=str)
    args = parser.parse_args()

    profile = json.load(open(args.filepath))
    durations = defaultdict(lambda: 0)

    for elem in profile:
        args = elem.get("args")
        if args:
            op = args.get("op_name")
            if op:
                dur = int(elem.get("dur"))
                durations[op] += dur

    table = [(op, dur / 1000.0) for op, dur in durations.items()]
    table.append(("*Total*", sum(dur for _, dur in table)))
    table = sorted(table, key=lambda x: x[1])

    print(tabulate(table, tablefmt="simple_outline", headers=["Op", "Duration [ms]"]))


if __name__ == "__main__":
    main()
