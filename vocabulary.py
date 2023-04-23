class Vocabulary:
    def __init__(self, sentences, max_vocab_size=None, min_word_freq=None):
        self.vocabulary = None
        self.word_to_index = {"<UNK>": 0}
        self.index_to_word = {0: "<UNK>"}
        self.build_vocab(sentences, max_vocab_size=max_vocab_size, min_word_freq=min_word_freq)

    def build_vocab(self, sentences, max_vocab_size, min_word_freq):
        vocabulary = {}
        for sentence in sentences:
            words = list(sentence)
            for word in words:
                if word not in vocabulary:
                    vocabulary[word] = 1
                else:
                    vocabulary[word] += 1

        if max_vocab_size is not None:
            vocabulary = dict(sorted(vocabulary.items(), key=lambda item: item[1], reverse=True)[:max_vocab_size])
        if min_word_freq is not None:
            vocabulary = {word: freq for word, freq in vocabulary.items() if freq >= min_word_freq}

        vocabulary = dict(sorted(vocabulary.items(), key=lambda item: item[1], reverse=True))

        for i, word in enumerate(vocabulary.keys()):
            self.word_to_index[word] = i + 1
            self.index_to_word[i + 1] = word

        self.vocabulary = vocabulary

    def __len__(self):
        return len(self.vocabulary)

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.word_to_index.get(item, None)
        elif isinstance(item, int):
            return self.index_to_word.get(item, None)
        else:
            raise KeyError("item must be either a word or an index")

    def get_frequency(self, word):
        return self.vocabulary.get(word, 0)
