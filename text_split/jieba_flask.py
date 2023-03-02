from flask import Flask, request
import jieba.analyse

app = Flask(__name__)


@app.route("/keywords", methods=["POST"])
def extract_keywords():
    text = request.json["text"]
    print("text", text)
    keywords = jieba.analyse.extract_tags(text, topK=10, withWeight=False, allowPOS=())
    print("keywords", keywords)
    return {"keywords:": keywords}


if __name__ == "__main__":
    app.run(port=9527)