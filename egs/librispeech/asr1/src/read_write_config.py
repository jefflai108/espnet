import shutil
import os

def write_lm_config():
    with open('conf/lm3.yaml', 'r') as f:
        content = f.readlines()
    content = {x.strip('\n').split(':')[0]:x.strip('\n').split(':')[1] for x in content}

    idx = 5
    for layer in [1, 2, 3, 4]:
        for unit in [64, 128, 256]:
            for dropout in [0.0, 0.1, 0.2, 0.3, 0.4]:
                out_f = 'conf/lm' + str(idx) + '.yaml'
                print(out_f)
                with open(out_f, 'w') as f:
                    for k, v in content.items():
                        if k == 'layer': f.write('layer: %d\n' % layer)
                        elif k == 'unit': f.write('unit: %d\n' % unit)
                        elif k == 'dropout-rate': f.write('dropout-rate: %f' % dropout)
                        else: f.write('%s:%s\n' % (k,v))
                idx += 1


def read_lm_results():

    smallest_pp = 100
    for file_id in range(4, 64):
        with open('exp/train_rnnlm_pytorch_lm' + str(file_id) + '_unigram600_ngpu1/train.log') as f:
             content = f.readlines()
        content = [x.strip('\n') for x in content]
        for idx, i in enumerate(content[::-1]):
            i = i.replace('\x1b[J]', '')
            if 'total' in i:
                break
        result_line = content[::-1][idx+1]
        current_ppl = float(result_line.split()[4])
        if current_ppl < smallest_pp:
            smallest_pp = current_ppl
            smallest_file_id = file_id
            print(smallest_file_id, smallest_pp)
    print(smallest_file_id, smallest_pp)


def write_train_config():
    with open('conf/train-debug3.yaml', 'r') as f:
        content = f.readlines()
    content = {x.strip('\n').split(':')[0]:':'.join(x.strip('\n').split(':')[1:]) for x in content}

    idx = 100
    for mtlalpha in [0.3]:
        for mtlalpha2 in [0.3]:
            for jointbeta1 in [1.0]:
                for jointbeta2 in [1.0]:
                        for dlayers2 in [1, 3, 6]:
                            for dunits2 in [512, 1024, 2048]:
                                for aheads2 in [4, 8]:
                                    out_f = 'conf/train-debug3.' + str(idx) + '.yaml'
                                    print(out_f)
                                    with open(out_f, 'w') as f:
                                        for k, v in content.items():
                                            if k == 'mtlalpha': f.write('mtlalpha: %.1f\n' % mtlalpha)
                                            elif k == 'mtlalpha2': f.write('mtlalpha2: %.1f\n' % mtlalpha2)
                                            elif k == 'jointbeta1': f.write('jointbeta1: %.1f\n' % jointbeta1)
                                            elif k == 'jointbeta2': f.write('jointbeta2: %.1f\n' % jointbeta2)
                                            elif k == 'dlayers2': f.write('dlayers2: %d\n' % dlayers2)
                                            elif k == 'dunits2': f.write('dunits2: %d\n' % dunits2)
                                            elif k == 'aheads2': f.write('aheads2: %d\n' % aheads2)
                                            else: f.write('%s:%s\n' % (k,v))
                                    idx += 1


if __name__ == '__main__':
    write_train_config()
