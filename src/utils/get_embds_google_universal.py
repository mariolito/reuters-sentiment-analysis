import tensorflow_hub as hub


class Embeddor(object):

    def __init__(self):
        self.embed = hub.load(
            "https://tfhub.dev/google/universal-sentence-encoder/4")

    def get_embds_google(self, list_text):
        return self.embed(list_text).numpy()
