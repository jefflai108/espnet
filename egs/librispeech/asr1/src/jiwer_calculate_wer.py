import json
from jiwer import wer

def wer_from_json(json_file):
    with open(json_file, encoding='utf-8') as jf:
        data = json.load(jf)
        reference  = []
        hypothesis = []
        for wav in data['utts']:
            # recognized text
            rec_text = data['utts'][wav]['output'][0]['rec_text']
            rec_text = rec_text.replace('▁', ' ')[:-len('<eos>')].lstrip()
            # text
            text = data['utts'][wav]['output'][0]['text']

            reference.append(text)
            hypothesis.append(rec_text)

        error = wer(reference, hypothesis)
        print(error * 100)

def read_espnet_results(results_wrd_md):
    with open(results_wrd_md, 'r+') as f:
        x = f.readline()
        while x.split() != 'Speaker':
            f.next()
            x = f.readline()
        print(x)



if __name__ == '__main__':
    #wer_from_json('decode/atis.test/result.json')
    #wer_from_json('decode/atis.train/result.json')
    #wer_from_json('decode/atis.valid/result.json')
    read_espnet_results('decode/atis.test/results.wrd.md')
