import numpy as np
import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification

emotion_names_jp = ['喜び', '悲しみ', '期待', '驚き', '怒り', '恐れ', '嫌悪', '信頼']

checkpoint = 'cl-tohoku/bert-base-japanese-whole-word-masking'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def np_softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

class Text_analyser():
    def __init__(self, root):
        model_path = os.path.join(root, "lang_analyser")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

    def analyse(self, text):
        self.model.eval()

        tokens = tokenizer(text, truncation=True, return_tensors='pt')
        tokens.to(self.model.device)
        preds = self.model(**tokens)
        prob = np_softmax(preds.logits.cpu().detach().numpy()[0])
        out_dict = {n: p for n, p in zip(emotion_names_jp, prob)}

        return out_dict