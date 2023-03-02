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

# jieba
import jieba.analyse


def extract_keywords_ckiptagger(text, top_k=20):
    # Tokenize the text
    word_sentence_list = ws([text])
    # POS tagging
    pos_sentence_list = pos(word_sentence_list)
    # NER tagging
    entity_sentence_list = ner(word_sentence_list, pos_sentence_list)
    # Extract Keywords
    keywords = []
    for entity in entity_sentence_list[0]:
        # if entity[2] in ["PERSON", "ORG", "GPE", "DATE"]:
        keywords.append(entity[3])
    return keywords[:top_k]


def extract_keywords_ckiptagger_key_words(text, key_words, top_k=20):
    # Tokenize the text
    word_sentence_list = ws([text])
    # POS tagging
    pos_sentence_list = pos(word_sentence_list)
    # NER tagging
    entity_sentence_list = ner(word_sentence_list, pos_sentence_list)
    # Extract Keywords
    keywords = []
    for entity in entity_sentence_list[0]:
        if entity[2] in [key_words]:
            keywords.append(entity[3])
    return keywords[:top_k]


# flask
app = Flask(__name__)


@app.route("/keywords", methods=["POST"])
def extract_keywords():
    text = request.json["text"]
    print("text", text)
    ckiptagger_keywords = extract_keywords_ckiptagger(text)
    date = extract_keywords_ckiptagger_key_words(text, "DATE")
    person = extract_keywords_ckiptagger_key_words(text, "PERSON")
    jieba_keywords_keywords = jieba.analyse.extract_tags(text, topK=10, withWeight=False, allowPOS=())
    return {
        "ckiptagger_keywords:": ckiptagger_keywords,
        "date:": date,
        "person:": person,
        "jieba_keywords_keywords": jieba_keywords_keywords
    }


if __name__ == "__main__":
    app.run(port=9527)