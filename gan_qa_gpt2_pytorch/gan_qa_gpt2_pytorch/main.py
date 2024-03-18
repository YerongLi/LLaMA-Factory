## main.py

from gan import GAN

def main():
    gan = GAN()
    gan.train()
    gan.evaluate()
    gan.generate_question_answer()

if __name__ == "__main__":
    main()
