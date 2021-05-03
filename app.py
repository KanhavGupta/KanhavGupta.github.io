from text_input import custom_text_check
import sys

sentiment = custom_text_check(sys.argv[1])
print(sys.argv[1] + ":  " + sentiment)
