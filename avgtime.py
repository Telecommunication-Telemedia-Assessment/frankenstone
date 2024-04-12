#!/usr/bin/env python3
import os
import sys
import timeit

if len(sys.argv) < 2:
    print("usage: ./avgtime.py <N> <CMD>")
    sys.exit(1)

number = int(sys.argv[1])
cmd = " ".join(sys.argv[2:])
time_values = []
for n in range(number):
    time_values.append(timeit.timeit(f"import os; os.system('{cmd}')", number=1))

import json
res = {"cmd": cmd, "avgtime": sum(time_values) / number, "values": time_values}
for i, a in enumerate(sys.argv[2:]):
    res["arg_" + str(i)] = a
print(json.dumps(res))

