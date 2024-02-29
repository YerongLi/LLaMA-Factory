from stanfordcorenlp import StanfordCoreNLP
import logging
import json
import tqdm

class StanfordNLP:
    def __init__(self, host='http://localhost', port=9000):
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
    # text = r'China on Wednesday issued a $50-billion list of U.S. goods  including soybeans and small aircraft for possible tariff hikes in an escalating technology dispute with Washington that companies worry could set back the global economic recovery.The country\'s tax agency gave no date for the 25 percent increase...'
    # ANNOTATE =  sNLP.annotate(text)
    # POS = sNLP.pos(text)
    # TOKENS = sNLP.word_tokenize(text)
    # NER = sNLP.ner(text)
    # PARSE = sNLP.parse(text)
    # DEP_PARSE = sNLP.dependency_parse(text)
    # Open the input file
    with open('summary.jsonl', 'r') as jsonl_file:
        # Open the output file
        with open('summary_w_key.json', 'w') as output_file:
            # Iterate through each line in the input file
            for line in tqdm.tqdm(jsonl_file):
                # Parse JSON from the line
                json_obj = json.loads(line)
                
                # Perform Named Entity Recognition (NER) using sNLP
                NER = sNLP.ner(json_obj['response'])
                
                # Extract non-'O' labeled items
                non_O_items = [item[0] for item in NER if item[1] != 'O']
                
                # Add the non-'O' items list to the JSON object
                json_obj['key'] = non_O_items
                
                # Write the modified JSON object to the output file
                print(non_O_items)
                print(len(non_O_items))
                output_file.write(json.dumps(json_obj) + '\n')
