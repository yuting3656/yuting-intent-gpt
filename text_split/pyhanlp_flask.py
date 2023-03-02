from flask import Flask, request
from pyhanlp import *
import json

app = Flask(__name__)


@app.route("/getKeywords", method=["post"])
def get_keywords():
    text = request.json["text"]
    seg = HanLP.segment(text)
    keywords = []
    for i in range(seg.size()):
        word = seg.get(i).word
        nature = seg.get(i).nature.toString()
        if nature.startswith("n") or nature.startswith("v"):
            keywords.append(word)
    return json.dumps(keywords)


if __name__ == "__main__":
    app.run()