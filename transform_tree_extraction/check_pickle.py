import pickle
import sys

if len(sys.argv) != 2:
    print("Usage: python check_pickle.py <image_time_file>")
    sys.exit(2)

pickle_file = sys.argv[1]

try:
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        print("Provided file does not contain a dictionary.")
        sys.exit(2)

    for key, value in data.items():
        if len(value) == 0:
            print(f"Key '{key}' does not have a corresponding transform.")
            sys.exit(1)

    # All transforms are non-empty
    sys.exit(0)

except Exception as e:
    print(f"Error: {e}")
    sys.exit(2)
