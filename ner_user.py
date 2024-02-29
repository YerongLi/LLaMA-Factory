from stanfordcorenlp import StanfordCoreNLP
import logging
import json
import tqdm
import os

class StanfordNLP:
    def __init__(self, host='http://localhost', port=13000):
        self.nlp = StanfordCoreNLP(host, port=port,
                                   timeout=30000 , quiet=True, logging_level=logging.DEBUG)
        self.props = {
            'annotators': 'tokenize,ssplit,pos,lemma,ner,parse,depparse,dcoref,relation,sentiment',
            'pipelineLanguage': 'en',
            'outputFormat': 'json'
        }

    def word_tokenize(self, sentence):
        return self.nlp.word_tokenize(sentence)

    def pos(self, sentence):
        return self.nlp.pos_tag(sentence)

    def ner(self, sentence):
        return self.nlp.ner(sentence)

    def parse(self, sentence):
        return self.nlp.parse(sentence)

    def dependency_parse(self, sentence):
        return self.nlp.dependency_parse(sentence)

    def annotate(self, sentence):
        return json.loads(self.nlp.annotate(sentence, properties=self.props))

    @staticmethod
    def tokens_to_dict(_tokens):
        tokens = defaultdict(dict)
        for token in _tokens:
            tokens[int(token['index'])] = {
                'word': token['word'],
                'lemma': token['lemma'],
                'pos': token['pos'],
                'ner': token['ner']
            }
        return tokens

if __name__ == '__main__':
    sNLP = StanfordNLP()

    # Remove the 'user4_w_key.jsonl' file if it exists
    if os.path.exists('user4_w_key.jsonl'):
        os.remove('user4_w_key.jsonl')

    with open('user4.jsonl', 'r') as count_file:
        total_lines = sum(1 for _ in count_file)

    # Open the input file
    cnt = 0
    with open('user4.jsonl', 'r') as jsonl_file:
        # Open the output file
        with open('user4_w_key.jsonl', 'a') as output_file:
            # Iterate through each line in the input file
            for line in tqdm.tqdm(jsonl_file, total=total_lines):
                # Parse JSON from the line
                json_obj = json.loads(line)
                
                # Perform Named Entity Recognition (NER) using sNLP
                if 'response' in json_obj:
                    NER = sNLP.ner(json_obj['response'])
                    cnt+= 1
                    print('NO response field')
                    print(json_obj.keys())
                else:
                    NER = []
                
                # Extract non-'O' labeled items
                non_O_items = {item[0]:'' for item in NER if item[1] != 'O'}  # Convert to set
                
                # Add the non-'O' items set to the JSON object
                json_obj['key'] = non_O_items
                
                # Write the modified JSON object to the output file
                output_file.write(json.dumps(json_obj) + '\n')
    print(f'Failed count {cnt}')