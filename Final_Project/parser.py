from create_features import features
from sklearn.externals import joblib
from collections import OrderedDict
from string import whitespace, punctuation
import os
import json

clf = joblib.load('frame_parser.pkl')
vec = joblib.load('feature_transformer.pkl')

def frames():
    header = ('word', 'lex', 'pos', 'gram', 'prev_gr', 'prev_lex', 'rel', 'pred_lemma')
    os.system("./mystem -icd --eng-gr --format json input.txt output.json")
    with open("output.json", "r", encoding="utf-8") as f:
        anas = json.load(f)
 #   anas = m.analyze(sentence)
    fr = OrderedDict()
    data = OrderedDict()
    for w in anas:
        word = w['text']
        try:
            gr = w['analysis'][0]['gr']
            lex = w['analysis'][0]['lex']
        except:
            gr, lex = None, None
        data[word] = [lex, gr, None, None, None, None]
    # vector = features(data)
    vectors = OrderedDict((k, v) for k,v in data.items() if k not in whitespace and k not in punctuation)
    for w in vectors:
        feats = [w] + [str(x) for x in vectors[w]][:-1]
        feats = dict(zip(header, feats))
        v = vec.transform(feats)
        role = clf.predict(v)[0]
        fr[w] = role
    return fr
    #vector = vec.transform(features)
    #return clf.predict(vector)[0][0]

if __name__ == "__main__":
    # clf = joblib.load('model.pkl')
    # vec = joblib.load('feature_transformer.pkl')

    # demo part
    with open ("input.txt", "w", encoding="utf-8") as f:
         phrase = input('Введите фразу: ')
         f.write(phrase)
    fr = frames()
    for f in fr:
        print(f, fr[f])
