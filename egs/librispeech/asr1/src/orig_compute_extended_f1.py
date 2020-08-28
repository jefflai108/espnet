import logging
import re
import sys

from utterance_aligner import UtteranceAligner


def get_indexs(utterance, span):
    tokens = re.split('\s+', utterance)
    span_tokens = re.split('\s+', span)
    for i in range(len(tokens) - len(span_tokens) + 1):
        match = True
        for j, span_token in enumerate(span_tokens):
            if span_token != tokens[i+j]:
                match = False
                break

        if match:
            return (i, i+len(span_tokens)-1)

    return None


def process_slot_types(string):
    return [re.split('\:', span)[0] for span in re.split('\;', string)]


def get_utterance_tp_fn_fp(aligner, transcription, hypothesis, transcription_slots, hypothesis_slots):
    transcription_slot_span_indexs = [
        get_indexs(transcription, transcription_slot) for transcription_slot in transcription_slots]
    hypothesis_slot_span_indexs = [
        get_indexs(hypothesis, hypothesis_slot) for hypothesis_slot in hypothesis_slots]

    transcription_slot_span_indexs = set([
        index for indexs in transcription_slot_span_indexs for index in range(indexs[0], indexs[1]+1)])
    hypothesis_slot_span_indexs = set([
        index for indexs in hypothesis_slot_span_indexs for index in range(indexs[0], indexs[1]+1)])

    _, alignment_trace = aligner.align_utterances(transcription, hypothesis)

    transcription_index, hypothesis_index = 0, 0
    TP, FN, FP = 0, 0, 0

    for alignment_symbol in alignment_trace:
        assert transcription_index <= len(re.split('\s+', transcription))
        assert hypothesis_index <= len(re.split('\s+', hypothesis))

        if alignment_symbol == UtteranceAligner.MATCH_SYM:
            if transcription_index in transcription_slot_span_indexs:
                if hypothesis_index in hypothesis_slot_span_indexs:
                    TP += 1
                else:
                    FN += 1
            else:
                if hypothesis_index in hypothesis_slot_span_indexs:
                    FP += 1
                else:
                    # slot = other
                    pass
            transcription_index += 1
            hypothesis_index += 1

        elif alignment_symbol == UtteranceAligner.DELETION_SYM:
            if transcription_index in transcription_slot_span_indexs:
                FN += 1
            else:
                # slot = other
                pass
            transcription_index += 1

        elif alignment_symbol == UtteranceAligner.INSERTION_SYM:
            if hypothesis_index in hypothesis_slot_span_indexs:
                FP += 1
            else:
                # slot = other
                pass
            hypothesis_index += 1

        elif alignment_symbol == UtteranceAligner.SUBSTITUTION_SYM:
            if transcription_index in transcription_slot_span_indexs:
                if hypothesis_index in hypothesis_slot_span_indexs:
                    FP += 1
                    FN += 1
                else:
                    FN += 1
            else:
                if hypothesis_index in hypothesis_slot_span_indexs:
                    FP += 1
                else:
                    # slot = other
                    pass
            transcription_index += 1
            hypothesis_index += 1

        else:
            raise ValueError

    return TP, FN, FP, alignment_trace


def process_test_case(test_case, aligner):
    transcription, hypothesis = test_case[0], test_case[1]
    transcription_slots = process_slot_types(test_case[2])
    hypothesis_slots = process_slot_types(test_case[3])

    TP, FN, FP, alignment_trace = get_utterance_tp_fn_fp(
        aligner, transcription, hypothesis, transcription_slots, hypothesis_slots)

    result = []
    if TP:
        result.append('{}:TP'.format(TP))
    if FN:
        result.append('{}:FN'.format(FN))
    if FP:
        result.append('{}:FP'.format(FP))

    result = ';'.join(result)

    return TP, FN, FP


def compute_char_edit_distance(token_1, token_2):
    aligner = UtteranceAligner()

    return aligner.align_utterances(
        ' '.join(list(token_1)), ' '.join(list(token_2)))[0]


def substitution_cost(alignment_matrix, i, j, tokens_1, tokens_2):
    """
    Ignore input and return 1 for all cases (insertion, deletion, substitution)
    """
    edit_distance = compute_char_edit_distance(tokens_1[i-1], tokens_2[j-1])
    #print(tokens_1[i-1], tokens_2[j-1], edit_distance/len(tokens_1[i-1]))

    if edit_distance/len(tokens_1[i-1]) < 0.7 or len(tokens_1[i-1]) < 4:
        return 1.5
    else:
        return 10

    #return 1 + min(edit_distance/len(tokens_1), 1)

def insert_delete_cost(alignment_matrix, i, j, tokens_1, tokens_2):
    edit_distance = compute_char_edit_distance(tokens_1[i-1], tokens_2[j-1])

    if edit_distance/len(tokens_1[i-1]) > 0.4:
        return 1
    else:
        return 10

    #return 1 - min(edit_distance/len(tokens_1), 1) * 0.75


def main():
    aligner = UtteranceAligner(
        substitution_cost=substitution_cost,
        insertion_cost=insert_delete_cost,
        deletion_cost=insert_delete_cost)

    test_case, TPs, FNs, FPs = [], 0, 0, 0
    utterance_files = sys.argv[1]
    with open(sys.argv[1]) as f:
        for line in f:
            line = line.strip()
            if not line:
                TP, FN, FP = process_test_case(test_case, aligner)
                TPs += TP
                FNs += FN
                FPs += FP
                test_case = []
            else:
                test_case.append(line)

    print('Extended SL F1: {}'.format(2*TPs/(2*TPs + FPs + FNs)))

if __name__ == '__main__':
    main()
