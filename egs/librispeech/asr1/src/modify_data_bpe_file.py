import json
from itertools import chain
import numpy as np
import copy
import utils.vocab_reader as vocab_reader

dataroot = '/home/ubuntu/tools/slot_filling_and_intent_detection_of_SLU/data/atis-2'
tag_vocab_dir = dataroot + '/vocab.slot2'
class_vocab_dir = dataroot + '/vocab.intent'

def write_to_dict(json_file1, matched_file1, json_file2, matched_file2, json_file3, matched_file3, \
                  out_file, tag_to_idx, class_to_idx):
    utts = []
    with open(json_file1, "rb") as f:
        tr_json = json.load(f)["utts"]
    utts.extend(list(tr_json.keys()))
    print(len(utts))

    matched_content = {}
    with open(matched_file1) as f:
        content = f.readlines()
    matched_content.update({x.strip('\n').split()[0]:' '.join(x.strip('\n').split()[1:]) for x in content})

    with open(out_file, 'w', encoding='utf8') as f:

        data = {}
        for id in utts:
            input  = copy.deepcopy(tr_json[id]['input'])
            output = copy.deepcopy(tr_json[id]['output'])
            tr_json[id]['output2'] = output
            tr_json[id]['output2'][0].pop('token')
            tr_json[id]['output2'][0].pop('tokenid')

            input     = tr_json[id]['input'][0]
            text      = tr_json[id]['output'][0]['text']
            token     = tr_json[id]['output'][0]['token']
            token_id  = tr_json[id]['output'][0]['tokenid']

            ref_item  = matched_content[id]
            ref_text  = ref_item.split(' EOS ')[0][4:]
            assert ref_text == text

            ref_slots = ' '.join(ref_item.split(' EOS ')[1].split()[1:-1])
            assert len(ref_slots.split()) == len(text.split())
            tag_seq = []
            for slot in ref_slots.split():
                if slot not in data.keys():
                    data[slot] = tag_to_idx[slot] + 1 # 0 is for <blank>

        for k, v in data.items():
            f.write('%s %d\n' % (k, int(v)))

def write_to_dict2(out_file):

    with open(tag_vocab_dir, 'r') as f:
        content = f.readlines()
    content = [x.strip('\n') for x in content]

    with open(out_file, 'w', encoding='utf8') as f:
        for idx, i in enumerate(content):
            f.write('%s %d\n' % (i, idx + 1 )) # 0 is for <blank>


def read_json(json_file, matched_file, out_file, tag_to_idx, class_to_idx):
    with open(json_file, "rb") as f:
        tr_json = json.load(f)["utts"]
    utts = list(tr_json.keys())
    print(len(utts))

    with open(matched_file) as f:
        content = f.readlines()
    matched_content = {x.strip('\n').split()[0]:' '.join(x.strip('\n').split()[1:]) for x in content}

    with open(out_file, 'w', encoding='utf8') as f:
        data = {}
        for id in utts:
            input  = copy.deepcopy(tr_json[id]['input'])
            output = copy.deepcopy(tr_json[id]['output'])
            tr_json[id]['output2'] = output
            tr_json[id]['output2'][0].pop('token')
            tr_json[id]['output2'][0].pop('tokenid')

            input     = tr_json[id]['input'][0]
            text      = tr_json[id]['output'][0]['text']
            token     = tr_json[id]['output'][0]['token']
            token_id  = tr_json[id]['output'][0]['tokenid']
            shape     = tr_json[id]['output'][0]['shape']

            token_select_id = [0] * len(token.split())
            for idx, i in enumerate(token.split()):
                if '▁' in i:
                    token_select_id[idx] = 1
            #print(token.split(), token_select_id)
            tr_json[id]['output'][0]['token_select_id'] = token_select_id

            ref_item  = matched_content[id]
            ref_text  = ref_item.split(' EOS ')[0][4:]
            assert ref_text == text

            ref_slots = ' '.join(ref_item.split(' EOS ')[1].split()[1:-1])
            assert len(ref_slots.split()) == len(text.split())
            tag_seq = []
            for slot in ref_slots.split():
                tag_seq.append(tag_to_idx[slot] if slot in tag_to_idx else (tag2idx['<unk>'], slot))
            tag_seq = ' '.join([str(x) for x in tag_seq])
            tr_json[id]['output2'][0]['name'] = 'target2'
            tr_json[id]['output2'][0]['name2'] = 'target3'
            tr_json[id]['output2'][0]['slots'] = ref_slots
            #tr_json[id]['output2'][0]['slots'] = token
            tr_json[id]['output2'][0]['slotsid'] = tag_seq
            #tr_json[id]['output2'][0]['slotsid'] = token_id
            tr_json[id]['output2'][0]['shape'] = [len(ref_slots.split()), 139+2] # +2 for <blank> and <eos>
            #tr_json[id]['output2'][0]['shape'] = shape

            ref_class = ref_item.split()[-1]
            tr_json[id]['output2'][0]['class'] = ref_class
            if '#' in ref_class: #'atis_flight#atis_airfare', 'atis_aircraft#atis_flight#atis_flight_no'
                class_label = class_to_idx[ref_class.split('#')[0]]
                tr_json[id]['output2'][0]['classid'] = (class_label, ref_class.split('#'))
            else:
                class_label = class_to_idx[ref_class]
                tr_json[id]['output2'][0]['classid'] = class_label

        data["utts"] = tr_json
        json.dump(data, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(",", ": "))


def read_json2(json_file, matched_file, out_file, tag_to_idx, class_to_idx):
    with open(json_file, "rb") as f:
        tr_json = json.load(f)["utts"]
    utts = list(tr_json.keys())
    print(len(utts))

    with open(matched_file) as f:
        content = f.readlines()
    matched_content = {x.strip('\n').split()[0]:' '.join(x.strip('\n').split()[1:]) for x in content}

    with open(out_file, 'w', encoding='utf8') as f:
        data = {}
        for id in utts:
            input  = copy.deepcopy(tr_json[id]['input'])
            output = copy.deepcopy(tr_json[id]['output'])
            tr_json[id]['output2'] = output
            #tr_json[id]['output2'][0].pop('token')
            #tr_json[id]['output2'][0].pop('tokenid')

            input     = tr_json[id]['input'][0]
            text      = tr_json[id]['output'][0]['text']
            token     = tr_json[id]['output'][0]['token']
            token_id  = tr_json[id]['output'][0]['tokenid']
            shape     = tr_json[id]['output'][0]['shape']

            ref_item  = matched_content[id]
            ref_text  = ref_item.split(' EOS ')[0][4:]
            assert ref_text == text

            ref_slots = ' '.join(ref_item.split(' EOS ')[1].split()[1:-1])
            assert len(ref_slots.split()) == len(text.split())
            tag_seq = []
            for slot in ref_slots.split():
                tag_seq.append(tag_to_idx[slot] + 1)
            tag_seq = ' '.join([str(x) for x in tag_seq])
            tr_json[id]['output2'][0]['name'] = 'target2'
            tr_json[id]['output2'][0]['name2'] = 'target3'
            tr_json[id]['output2'][0]['slots'] = ref_slots
            #tr_json[id]['output2'][0]['slots'] = token
            tr_json[id]['output2'][0]['slotsid'] = tag_seq
            #tr_json[id]['output2'][0]['slotsid'] = token_id
            tr_json[id]['output2'][0]['shape'] = [len(ref_slots.split()), 126+2]
            #tr_json[id]['output2'][0]['shape'] = shape

            print(text)
            print(token)
            print(ref_slots)
            print(len(token.split()), len(ref_slots.split()))
            continue

            ref_class = ref_item.split()[-1]
            if '#' in ref_class: #'atis_flight#atis_airfare', 'atis_aircraft#atis_flight#atis_flight_no'
                ref_class = ref_class.split('#')[0]
            class_label = class_to_idx[ref_class]
            tr_json[id]['output2'][0]['class'] = ref_class
            tr_json[id]['output2'][0]['classid'] = class_label

        data["utts"] = tr_json
        json.dump(data, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(",", ": "))

def load_slot_and_class_idx(testing=False):
    if not testing:
        tag_to_idx, idx_to_tag = vocab_reader.read_vocab_file(tag_vocab_dir, bos_eos=False, no_pad=False, no_unk=False)
        class_to_idx, idx_to_class = vocab_reader.read_vocab_file(class_vocab_dir, bos_eos=False, no_pad=False, no_unk=False)
    else:
        tag_to_idx, idx_to_tag = vocab_reader.read_vocab_file(opt.read_vocab+'.tag', bos_eos=False, no_pad=True, no_unk=True)
        class_to_idx, idx_to_class = vocab_reader.read_vocab_file(opt.read_vocab+'.class', bos_eos=False, no_pad=True, no_unk=True)

    print("Vocab size: %s %s" % (len(tag_to_idx), len(class_to_idx)))
    print(class_to_idx)
    return (tag_to_idx, idx_to_tag), (class_to_idx, idx_to_class)

def _return_selected_indices(text, subword_token):
    text = [x.strip('\"') for x in text]
    subword_token = [x.strip('\"') for x in subword_token]
    print(text)
    print(subword_token)

    if len(text) == len(subword_token):
        return [1] * len(subword_token)

    select_indices = [0] * len(subword_token)
    text_pnt = 0
    for subword_pnt, subword in enumerate(subword_token):
        print(subword_pnt, subword, text_pnt)
        if subword in text[text_pnt] or subword[1:] in text[text_pnt]:
            text_pnt += 1
            select_indices[subword_pnt] = 1
        else:
            print('not in')

    return select_indices

def _convert_text_to_subword(x):
    ini_list = [x.split() for x in x.split('▁')[1:]]
    flatten_list = [j for sub in ini_list for j in sub]
    return flatten_list

if __name__ == '__main__':
    (tag_to_idx, idx_to_tag), (class_to_idx, idx_to_class) = load_slot_and_class_idx()
    #json_file1 = 'dump/atis.train/deltafalse/data_bpe1000.json'
    #matched_file1 = '/home/ubuntu/tools/atis-mapping/matched-atis-2.train.w-intent.iob'
    #out_file1 = 'dump/atis.train/deltafalse/data_bpe1000_slots2.json'
    #json_file2 = 'dump/atis.valid/deltafalse/data_bpe1000.json'
    #matched_file2 = '/home/ubuntu/tools/atis-mapping/matched-atis-2.dev.w-intent.iob'
    #out_file2 = 'dump/atis.valid/deltafalse/data_bpe1000_slots2.json'
    #json_file3 = 'dump/atis.test3/deltafalse/data_bpe1000.json'
    #matched_file3 = '/home/ubuntu/tools/atis-mapping/matched-atis.test.w-intent.iob'
    #out_file3 = 'dump/atis.test3/deltafalse/data_bpe1000_slots2.json'

    # for noisy atis
    json_file1_n = 'dump/atis.train_noise/deltafalse/data_bpe1000.json'
    matched_file1_n = '/home/ubuntu/tools/atis-mapping/matched-atis-2.train_noise.w-intent.iob'
    out_file1_n = 'dump/atis.train_noise/deltafalse/data_bpe1000_slots2.json'
    json_file2_n = 'dump/atis.valid_noise/deltafalse/data_bpe1000.json'
    matched_file2_n = '/home/ubuntu/tools/atis-mapping/matched-atis-2.dev_noise.w-intent.iob'
    out_file2_n = 'dump/atis.valid_noise/deltafalse/data_bpe1000_slots2.json'
    json_file3_n = 'dump/atis.test3_noise/deltafalse/data_bpe1000.json'
    matched_file3_n = '/home/ubuntu/tools/atis-mapping/matched-atis.test3_noise.w-intent.iob'
    out_file3_n = 'dump/atis.test3_noise/deltafalse/data_bpe1000_slots2.json'

    read_json(json_file1_n, matched_file1_n, out_file1_n, tag_to_idx, class_to_idx)
    read_json(json_file2_n, matched_file2_n, out_file2_n, tag_to_idx, class_to_idx)
    read_json(json_file3_n, matched_file3_n, out_file3_n, tag_to_idx, class_to_idx)

    #read_json2(json_file1, matched_file1, out_file4, tag_to_idx, class_to_idx)
    #read_json2(json_file2, matched_file2, out_file2, tag_to_idx, class_to_idx)
    #read_json2(json_file3, matched_file3, out_file3, tag_to_idx, class_to_idx)

    out_dict = 'data/lang_char/atis.train_bpe1000_units2.txt'
    #write_to_dict(json_file1, matched_file1, json_file2, matched_file2,
    #              json_file3, matched_file3, out_dict, tag_to_idx, class_to_idx)
    write_to_dict2(out_dict)

