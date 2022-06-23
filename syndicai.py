import json
import spacy

class PythonPredictor:
    def __init__(self, config):
        # load spacy model
        self.nlp = spacy.load('en_core_web_sm')
      
    def predict(self, payload):
        print(payload)
        #data = json.loads(payload)
        payload_text = payload["text"]
        preprocessed_text = payload_text.strip().replace("\n","")
        
        # load data
        sentence = "Apple is looking at buying U.K. startup for $1 billion"
        doc = self.nlp(preprocessed_text)

        # print entities
        for ent in doc.ents:
            print(ent.text, ent.start_char, ent.end_char, ent.label_)

        return {'original': payload_text, 'ner': docs.ents}
      
