# Utility functions for the NER model.

def read_conll(path):
    """Reads a CoNLL file and returns a list of sentences and their corresponding labels.

    Args:
        path (str): Path to the CoNLL file.

    Returns:
        sents: List of sentences.
        labels: List of labels for each sentence.
    """
    sents = []
    labels = []

    with open(path, 'r') as f:
        raw_sents = f.read().split("\n\n")
    
    for raw_sent in raw_sents:
        text = []
        label = []
        for line in raw_sent.split("\n"):
            if line != "":
                line = line.split("\t")
                text.append(line[0])
                label.append(line[1])
        sents.append(text)
        labels.append(label)

    return sents, labels