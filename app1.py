from emotion_detection import custom_audio_check
import sys

sentiment = custom_audio_check(sys.argv[1])
print(sentiment)
