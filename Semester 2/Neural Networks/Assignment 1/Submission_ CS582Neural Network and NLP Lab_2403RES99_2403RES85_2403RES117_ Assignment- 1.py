import pandas as pd
from collections import defaultdict
import ast

file_path = "NER_Dataset.csv"
df = pd.read_csv(file_path)

df["Word"] = df["Word"].apply(ast.literal_eval)
df["POS"] = df["POS"].apply(ast.literal_eval)
df["Tag"] = df["Tag"].apply(ast.literal_eval)


flattened_data = []
for _, row in df.iterrows():
    words = row["Word"]
    tags = row["Tag"]
    pos_tags = row["POS"]
    for word, pos, tag in zip(words, pos_tags, tags):
        flattened_data.append((word, pos, tag))


flat_df = pd.DataFrame(flattened_data, columns=["Word", "POS", "Tag"])


pos_transition_probs = defaultdict(lambda: defaultdict(int))
ner_transition_probs = defaultdict(lambda: defaultdict(int))
pos_emission_probs = defaultdict(lambda: defaultdict(int))
ner_emission_probs = defaultdict(lambda: defaultdict(int))
pos_counts = defaultdict(int)
ner_tag_counts = defaultdict(int)


prev_pos = "<START>"
prev_ner_tag = "<START>"


for _, row in flat_df.iterrows():
    word, pos, tag = row["Word"], row["POS"], row["Tag"]

    pos_transition_probs[prev_pos][pos] += 1
    pos_emission_probs[pos][word] += 1
    pos_counts[pos] += 1

    ner_transition_probs[prev_ner_tag][tag] += 1
    ner_emission_probs[tag][word] += 1
    ner_tag_counts[tag] += 1

    prev_pos = pos
    prev_ner_tag = tag


pos_transition_probs[prev_pos]["<END>"] += 1
ner_transition_probs[prev_ner_tag]["<END>"] += 1


def normalize_counts(counts):
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


pos_transition_probs = {k: normalize_counts(v) for k, v in pos_transition_probs.items()}
pos_emission_probs = {k: normalize_counts(v) for k, v in pos_emission_probs.items()}
ner_transition_probs = {k: normalize_counts(v) for k, v in ner_transition_probs.items()}
ner_emission_probs = {k: normalize_counts(v) for k, v in ner_emission_probs.items()}


def viterbi_algorithm_joint(
    sentence,
    pos_transition_probs,
    pos_emission_probs,
    pos_counts,
    ner_transition_probs,
    ner_emission_probs,
    ner_tag_counts,
):
    pos_tags = list(pos_counts.keys())
    ner_tags = list(ner_tag_counts.keys())

    V = [{}]
    path = {}

    for pos in pos_tags:
        for ner_tag in ner_tags:
            V[0][(pos, ner_tag)] = (
                pos_transition_probs["<START>"].get(pos, 0)
                * ner_transition_probs["<START>"].get(ner_tag, 0)
                * pos_emission_probs[pos].get(sentence[0], 1e-6)
                * ner_emission_probs[ner_tag].get(sentence[0], 1e-6)
            )
            path[(pos, ner_tag)] = [(sentence[0], pos, ner_tag)]

    for t in range(1, len(sentence)):
        V.append({})
        new_path = {}

        for pos in pos_tags:
            for ner_tag in ner_tags:

                (prob, (prev_pos, prev_ner_tag)) = max(
                    (
                        V[t - 1][(prev_pos, prev_ner_tag)]
                        * pos_transition_probs[prev_pos].get(pos, 0)
                        * ner_transition_probs[prev_ner_tag].get(ner_tag, 0)
                        * pos_emission_probs[pos].get(sentence[t], 1e-6)
                        * ner_emission_probs[ner_tag].get(sentence[t], 1e-6),
                        (prev_pos, prev_ner_tag),
                    )
                    for prev_pos in pos_tags
                    for prev_ner_tag in ner_tags
                    if (prev_pos, prev_ner_tag) in V[t - 1]
                )

                V[t][(pos, ner_tag)] = prob
                new_path[(pos, ner_tag)] = path[(prev_pos, prev_ner_tag)] + [
                    (sentence[t], pos, ner_tag)
                ]

        path = new_path

    n = len(sentence) - 1
    (prob, (pos, ner_tag)) = max(
        (V[n][(pos, ner_tag)], (pos, ner_tag))
        for pos in pos_tags
        for ner_tag in ner_tags
        if (pos, ner_tag) in V[n]
    )

    return path[(pos, ner_tag)]


sample_sentence = [
    "Thousands",
    "of",
    "demonstrators",
    "have",
    "marched",
    "through",
    "London",
    "Saturday",
    "An",
]


predicted_tuples = viterbi_algorithm_joint(
    sample_sentence,
    pos_transition_probs,
    pos_emission_probs,
    pos_counts,
    ner_transition_probs,
    ner_emission_probs,
    ner_tag_counts,
)
print(predicted_tuples)



def calculate_pos_accuracy(actual_tags, predicted_tags, sentence_words):
    """
    Calculate the accuracy of the model for specific POS tags (NNS and NNP).
    
    :param actual_tags: The list of actual POS tags from the dataset.
    :param predicted_tags: The list of predicted POS tags from the Viterbi algorithm.
    :param sentence_words: List of words from the sentence.
    :return: Accuracy for NNS and NNP tags only.
    """
    total_nns_nnp = 0
    correct_nns_nnp = 0
    
    for actual, predicted, word in zip(actual_tags, predicted_tags, sentence_words):
        if actual in ['NNS', 'NNP']:  # Focus only on NNS and NNP tags
            total_nns_nnp += 1
            if actual == predicted:
                correct_nns_nnp += 1

    if total_nns_nnp == 0:
        return None  # No NNS or NNP tags in the sentence to calculate accuracy
    
    accuracy = correct_nns_nnp / total_nns_nnp
    return accuracy


def evaluate_accuracy_for_nns_nnp(df, viterbi_algorithm_joint, pos_transition_probs, pos_emission_probs, pos_counts, ner_transition_probs, ner_emission_probs, ner_tag_counts):
    """
    Evaluate the accuracy of the model for NNS and NNP POS tags.
    
    :param df: The dataframe containing sentences, POS tags, and NER tags.
    :param viterbi_algorithm_joint: The Viterbi algorithm function for POS and NER tagging.
    :param pos_transition_probs: POS transition probabilities.
    :param pos_emission_probs: POS emission probabilities.
    :param pos_counts: POS counts.
    :param ner_transition_probs: NER transition probabilities.
    :param ner_emission_probs: NER emission probabilities.
    :param ner_tag_counts: NER tag counts.
    :return: Overall accuracy for NNS and NNP tags across all sentences.
    """
    total_accuracy = 0
    count = 0
    
    for _, row in df.head(10).iterrows():
        sentence_words = row["Word"]  # Directly use the list of words
        actual_pos_tags = row["POS"]  # Directly use the list of POS tags
        
        # Get the predicted tags using the Viterbi algorithm
        predicted_tuples = viterbi_algorithm_joint(
            sentence_words,
            pos_transition_probs,
            pos_emission_probs,
            pos_counts,
            ner_transition_probs,
            ner_emission_probs,
            ner_tag_counts,
        )
        predicted_pos_tags = [pos for _, pos, _ in predicted_tuples]
        
        accuracy = calculate_pos_accuracy(actual_pos_tags, predicted_pos_tags, sentence_words)
        
        if accuracy is not None:
            total_accuracy += accuracy
            count += 1
    
    overall_accuracy = total_accuracy / count if count > 0 else 0
    return overall_accuracy

overall_accuracy = evaluate_accuracy_for_nns_nnp(
    df,
    viterbi_algorithm_joint,
    pos_transition_probs,
    pos_emission_probs,
    pos_counts,
    ner_transition_probs,
    ner_emission_probs,
    ner_tag_counts
)
print(f"Overall accuracy for NNS and NNP tags: {overall_accuracy}")