from SimUGANSpeech.tf_session.mnistsession import MnistSession

if __name__ == "__main__":
    mnist_session = MnistSession()
    mnist_session.train(10)


