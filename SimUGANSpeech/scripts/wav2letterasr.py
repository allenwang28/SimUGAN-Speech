from SimUGANSpeech.tf_session.wav2lettersession import Wav2LetterSession

if __name__ == "__main__":
    asr_session = Wav2LetterSession()
    asr_session.train(1)
