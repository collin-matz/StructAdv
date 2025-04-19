import random
import jsonlines
from tqdm import tqdm
import spacy
import copy

# Load spaCy model
spacy.prefer_gpu()
nlp = spacy.load("en_core_web_sm")


# mappings from spacy and Penn Treebank POS tags
PRN = "PRON"
DT = "DET"
NN = "NOUN"
VBZ = "VERB"
JJ = "ADJ"
PRP = "PRON"
VBP = "VERB"
CC = "CONJ"
IN = "ADP"
VBD = "VERB"
NNP = "PROPN"
TO = "NOUN"
VB = "ADP"
NNPS = "NOUN"
CD = "NUM"
NNS = "NOUN"
MD = "VERB"

# here, we define valid structures. These are very general, but should hopefully
# give us sentence structures that are close enough to full sentences to trick
# the model
VALID_POS_STRUCTURES = [
    [DT, NN, VBZ, JJ],
    [NNP, VBZ, DT, JJ, NN, IN, NN],
    [PRP, VBD, DT, NN, TO, VB, PRP],
    [IN, DT, NN, VBZ, JJ, CC, JJ],
    [DT, NN, VBD, IN, DT, JJ, NN],
    [DT, NN, VBD, CD, JJ, NNS],
    [PRN, NN, VBZ, CD, JJ, NN, IN, DT, NN],
    [DT, NN, IN, CD, NNS, VBP, JJ],
    [CD, NNS, VBD, IN, DT, JJ, NN],
    [DT, NN, VBD, PRP, CD, NNS, TO, VB],
    [PRP, MD, VB, CD, JJ, NN, IN, NN],
    [CD, JJ, NNS, VBZ, JJ, CC, JJ],
]


def generate(qa_pair):
    # make a deep copy of the qa pair, so we don't modify the original
    qa_pair = copy.deepcopy(qa_pair)

    # isolate parts of qa_pairs
    question = qa_pair["question"]

    # convert question to nlp to get nouns, verbs, and adjectives from it
    doc = nlp(question)
    tokens = {
        "NOUN": [],
        "VERB": [],
        "ADJ": [],
        "PROPN": [],
        # pre-populate tags that we can reuse
        "NUM": [1, 2, 3, 1997, 1998, 1999],
        "PRON": ["they"],
        "DET": ["the", "a", "an"],
        "CONJ": ["and", "or", "but"],
        "ADP": ["in", "to", "during"],
    }

    # sort the words in our question
    for token in doc:
        if token.pos_ in tokens:
            tokens[token.pos_].append(token.text)

    # now that we have sorted the POS tags, we can generate a sentence by
    # picking the valid POS structure that closest matches our question.
    # we can do this by iterating over all the structs, generating a sentence
    # for each, and checking how full it is. the most full sentence will be used
    candidate_sentences = []
    struct_selections = random.choices(VALID_POS_STRUCTURES, k=len(VALID_POS_STRUCTURES))
    for pos_struct in struct_selections:
        candidate_sent = []
        for pos_tag in pos_struct:
            word_from_question = tokens[pos_tag]

            # skip if we don't have any words for this type
            if len(word_from_question) == 0:
                continue

            # if we do have words for this type, randomly
            # select one
            word = random.choice(word_from_question)
            candidate_sent.append(word)

        # once we have built our sentence, append it to our total candidate
        # sentences
        candidate_sentences.append(candidate_sent)

    # now, we want to pick the largest sentence that was generated. we assume
    # the more tokens that matched, the better the content of the sentence
    # will be
    sentence_to_use = candidate_sentences[0]
    max_len = len(sentence_to_use)
    for sent in candidate_sentences[1:]:
        if len(sent) > max_len:
            max_len = len(sent)
            sentence_to_use = sent

    # pick a random count of these key words to add to a random sentence
    punctuation = random.choice(['.', '!'])
    bad_sentence = " "
    for word in sentence_to_use[:-1]:
        bad_sentence += str(word) + " "
    bad_sentence += str(sentence_to_use[-1]) + punctuation
    
    qa_pair["context"] += bad_sentence
    
    return qa_pair


def process(dataset_json, output):
    # go through the entire squad training and eval dataset and add adversarial
    # examples using AddSent

    data = []
    adversarial_data = []
    adversarial_counts = []

    with jsonlines.open(dataset_json, mode='r') as reader:
        for qa_pair in tqdm(reader):

            data.append(qa_pair)

            # between 1 and 5 new examples will be added for each question, to avoid
            # biasing towards the same number of adversarial examples for each question
            number_of_adversarial_examples_to_add = random.randint(1,3)
            adversarial_counts.append(number_of_adversarial_examples_to_add)

            # generate uniform random adversarial sentences
            for _ in range(number_of_adversarial_examples_to_add):
                sent = generate(qa_pair)
                data.append(sent)
                adversarial_data.append(sent)
            
    # rewrite these newly generated pairs to another file
    with jsonlines.open(output, mode='w') as writer:
        writer.write_all(data)

    # write adversarial examples to a file
    with jsonlines.open("adversarial_validate_examples.json", mode='w') as writer:
        writer.write_all(adversarial_data)

    # write the number of adversarial examples added to a file
    with open("adversarial_validate_count.txt", 'w') as file:
        file.write('\n'.join(str(count) for count in adversarial_counts))

if __name__ == "__main__":
    process("squad_validate.json", "squad_adversarial_validate.json")
    