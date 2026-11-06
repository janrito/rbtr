"""Run the greeter from the command line."""

import sys

sys.path.insert(0, ".")
OPTIONS["verbose"] = True

try:
    import readline
except ImportError:
    readline = None

for name in sys.argv[1:]:
    print(name)

with open("greeting.log", "a") as log:
    log.write("ran\n")

del readline
assert sys.argv

if __name__ == "__main__":
    print("greeter")
