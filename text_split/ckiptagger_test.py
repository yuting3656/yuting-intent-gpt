from flask import Flask, request
from ckiptagger import data_utils
from ckiptagger import WS, POS, NER
import os
if os.path.isdir("./data"):
    print("yo data exists")
else:
    data_utils.download_data_gdown("./")

ws = WS("./data")
pos = POS("./data")
ner = NER("./data")


def extract_keywords_ckiptagger(text, top_k=20):
    # Tokenize the text
    word_sentence_list = ws([text])
    print("word_sentence_list", word_sentence_list)
    # POS tagging
    pos_sentence_list = pos(word_sentence_list)
    print("pos_sentence_list", pos_sentence_list)
    # NER tagging
    entity_sentence_list = ner(word_sentence_list, pos_sentence_list)
    print("entity_sentence_list", entity_sentence_list)
    # Extract Keywords
    keywords = []
    for entity in entity_sentence_list[0]:
        keywords.append(entity[3])
    return keywords[:top_k]


if __name__ == "__main__":
    ｗ = extract_keywords_ckiptagger("在想吼 如果有這麼厲害的話 那不就一下子就升天了唷唷～？")
    print("w", w)