import torch
import json 
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config


class PythonPredictor:
    def __init__(self, config):
      model = T5ForConditionalGeneration.from_pretrained('t5-small')
      tokenizer = T5Tokenizer.from_pretrained('t5-small')
      
    def predict(self, payload):
        conditioned_tokens = self.tokenizer.encode(payload["text"]) + [generator.END_OF_TEXT]
        prediction = generator.generate(self.model, conditioned_tokens, self.device)
        return self.tokenizer.decode(prediction)
      
        preprocessed_text = payload["text"].strip().replace("\n","")
        t5_prepared_Text = "summarize: " + preprocessed_text

        tokenized_text = self.tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)

        # summmarize 
        summary_ids = self.model.generate(tokenized_text,
                                            num_beams=4,
                                            no_repeat_ngram_size=2,
                                            min_length=30,
                                            max_length=100,
                                            early_stopping=True)

        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)    
        return {'original': payload.text, 'summary': summary}
      
