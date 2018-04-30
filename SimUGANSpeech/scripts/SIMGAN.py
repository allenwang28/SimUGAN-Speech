from SimUGANSpeech.tf_session.simgan_session import SimGANSession

if __name__ == "__main__":
    simgan_session = SimGANSession()
    simgan_session.train(10)