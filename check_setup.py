import src.train as train


def main():
    print('Hello from voice-gender-type-classification!')
    print('Your device supports [', train.device_detected(), '] in training.')

if __name__ == "__main__":
    main()
