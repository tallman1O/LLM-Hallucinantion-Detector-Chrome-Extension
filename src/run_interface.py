from src.interface import run_interface
from pprint import pprint

if __name__ == "__main__":
    text = input("Paste LLM output:\n\n")
    out = run_interface(text)
    pprint(out)