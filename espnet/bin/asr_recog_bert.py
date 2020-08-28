#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""End-to-end speech recognition model decoding script."""

import configargparse
import logging
import os
import random
import sys

import numpy as np

from espnet.utils.cli_utils import strtobool

# NOTE: you need this func to generate our sphinx doc


def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Transcribe text from speech using "
        "a speech recognition model on one CPU or GPU",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    parser.add("--config", is_config_file=True, help="Config file path")
    parser.add(
        "--config2",
        is_config_file=True,
        help="Second config file path that overwrites the settings in `--config`",
    )
    parser.add(
        "--config3",
        is_config_file=True,
        help="Third config file path that overwrites the settings "
        "in `--config` and `--config2`",
    )

    parser.add_argument("--ngpu", type=int, default=0, help="Number of GPUs")
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32", "float64"),
        default="float32",
        help="Float precision (only available in --api v2)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="chainer",
        choices=["chainer", "pytorch"],
        help="Backend library",
    )
    parser.add_argument("--debugmode", type=int, default=1, help="Debugmode")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--verbose", "-V", type=int, default=1, help="Verbose option")
    parser.add_argument(
        "--batchsize",
        type=int,
        default=1,
        help="Batch size for beam search (0: means no batch processing)",
    )
    parser.add_argument(
        "--preprocess-conf",
        type=str,
        default=None,
        help="The configuration file for the pre-processing",
    )
    parser.add_argument(
        "--api",
        default="v1",
        choices=["v1", "v2"],
        help="Beam search APIs "
        "v1: Default API. It only supports the ASRInterface.recognize method "
        "and DefaultRNNLM. "
        "v2: Experimental API. It supports any models that implements ScorerInterface.",
    )
    # task related
    parser.add_argument(
        "--recog-json", type=str, help="Filename of recognition data (json)"
    )
    parser.add_argument(
        "--result-label",
        type=str,
        required=True,
        help="Filename of result label data (json)",
    )
    # model (parameter) related
    parser.add_argument(
        "--model", type=str, required=True, help="Model file parameters to read"
    )
    parser.add_argument(
        "--model-conf", type=str, default=None, help="Model config file"
    )
    parser.add_argument(
        "--num-spkrs",
        type=int,
        default=1,
        choices=[1, 2],
        help="Number of speakers in the speech",
    )
    parser.add_argument(
        "--num-encs", default=1, type=int, help="Number of encoders in the model."
    )
    # search related
    parser.add_argument("--nbest", type=int, default=1, help="Output N-best hypotheses")
    parser.add_argument("--beam-size", type=int, default=1, help="Beam size")
    parser.add_argument("--penalty", type=float, default=0.0, help="Incertion penalty")
    parser.add_argument(
        "--maxlenratio",
        type=float,
        default=0.0,
        help="""Input length ratio to obtain max output length.
                        If maxlenratio=0.0 (default), it uses a end-detect function
                        to automatically find maximum hypothesis lengths""",
    )
    parser.add_argument(
        "--minlenratio",
        type=float,
        default=0.0,
        help="Input length ratio to obtain min output length",
    )
    parser.add_argument(
        "--ctc-weight", type=float, default=0.0, help="CTC weight in joint decoding"
    )
    parser.add_argument(
        "--weights-ctc-dec",
        type=float,
        action="append",
        help="ctc weight assigned to each encoder during decoding."
        "[in multi-encoder mode only]",
    )
    parser.add_argument(
        "--ctc-window-margin",
        type=int,
        default=0,
        help="""Use CTC window with margin parameter to accelerate
                        CTC/attention decoding especially on GPU. Smaller magin
                        makes decoding faster, but may increase search errors.
                        If margin=0 (default), this function is disabled""",
    )
    # transducer related
    parser.add_argument(
        "--score-norm-transducer",
        type=strtobool,
        nargs="?",
        default=True,
        help="Normalize transducer scores by length",
    )
    # rnnlm related
    parser.add_argument(
        "--rnnlm", type=str, default=None, help="RNNLM model file to read"
    )
    parser.add_argument(
        "--rnnlm-conf", type=str, default=None, help="RNNLM model config file to read"
    )
    parser.add_argument(
        "--word-rnnlm", type=str, default=None, help="Word RNNLM model file to read"
    )
    parser.add_argument(
        "--word-rnnlm-conf",
        type=str,
        default=None,
        help="Word RNNLM model config file to read",
    )
    parser.add_argument("--word-dict", type=str, default=None, help="Word list to read")
    parser.add_argument("--lm-weight", type=float, default=0.1, help="RNNLM weight")
    # ngram related
    parser.add_argument(
        "--ngram-model", type=str, default=None, help="ngram model file to read"
    )
    parser.add_argument("--ngram-weight", type=float, default=0.1, help="ngram weight")
    parser.add_argument(
        "--ngram-scorer",
        type=str,
        default="part",
        choices=("full", "part"),
        help="""if the ngram is set as a part scorer, similar with CTC scorer,
                ngram scorer only scores topK hypethesis.
                if the ngram is set as full scorer, ngram scorer scores all hypthesis
                the decoding speed of part scorer is musch faster than full one""",
    )
    # streaming related
    parser.add_argument(
        "--streaming-mode",
        type=str,
        default=None,
        choices=["window", "segment"],
        help="""Use streaming recognizer for inference.
                        `--batchsize` must be set to 0 to enable this mode""",
    )
    parser.add_argument("--streaming-window", type=int, default=10, help="Window size")
    parser.add_argument(
        "--streaming-min-blank-dur",
        type=int,
        default=10,
        help="Minimum blank duration threshold",
    )
    parser.add_argument(
        "--streaming-onset-margin", type=int, default=1, help="Onset margin"
    )
    parser.add_argument(
        "--streaming-offset-margin", type=int, default=1, help="Offset margin"
    )

    # slots related below
    parser.add_argument('--slots_task_st', required=True, help='slot filling task: NN | NN_crf')
    parser.add_argument('--slots_task_sc', required=True, help='intent detection task: none | CLS | max | CLS_max | CLS_decoder')
    parser.add_argument('--slots_sc_type', default='single_cls_CE', help='single_cls_CE | multi_cls_BCE')
    parser.add_argument('--slots_st_weight', type=float, default=0.5, help='loss weight for slot tagging task, ranging from 0 to 1.')

    parser.add_argument('--slots_save_model', default='model', help='save model to this file')
    #parser.add_argument('--slots_mini_word_freq', type=int, default=2, help='mini_word_freq in the training data')
    #parser.add_argument('--slots_word_lowercase', action='store_true', help='word lowercase')
    parser.add_argument('--slots_bos_eos', action='store_true', help='Whether to add <s> and </s> to the input sentence (default is not)')
    parser.add_argument('--slots_save_vocab', default='vocab', help='save vocab to this file')
    parser.add_argument('--slots_noStdout', action='store_true', help='Only log to a file; no stdout')

    parser.add_argument('--slots_testing', action='store_true', help='Only test your model (default is training && testing)')
    parser.add_argument('--slots_read_model', required=False, help='Online test: read model from this file')
    parser.add_argument('--slots_read_vocab', required=False, help='Online test: read input vocab from this file')
    parser.add_argument('--slots_out_path', required=False, help='Online test: out_path')

    #parser.add_argument('--slots_pretrained_model_type', required=True, help='bert, xlnet')
    parser.add_argument('--slots_pretrained_model_name', required=True, help='bert-base-uncased, bert-base-cased, bert-large-uncased, bert-large-cased, bert-base-multilingual-cased, bert-base-chinese; xlnet-base-cased, xlnet-large-cased')

    #parser.add_argument('--slots_lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--slots_dropout', type=float, default=0., help='dropout rate at each non-recurrent layer')
    parser.add_argument('--slots_batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--slots_test_batchSize', type=int, default=0, help='input batch size in decoding')
    parser.add_argument('--slots_gradient_accumulation_steps', type=int, default=1, help='Number of updates steps to accumulate before performing a backward/update pass.')
    parser.add_argument('--slots_init_weight', type=float, default=0.2, help='all weights will be set to [-init_weight, init_weight] during initialization')
    #parser.add_argument('--slots_max_norm', type=float, default=5, help="threshold of gradient clipping (2-norm)")
    #parser.add_argument('--slots_max_epoch', type=int, default=50, help='max number of epochs to train for')
    #parser.add_argument('--slots_experiment', default='exp', help='Where to store samples and models')
    #parser.add_argument('--slots_optim', default='bertadam', help='choose an optimizer')
    #parser.add_argument('--slots_warmup_proportion', type=float, default=0.1, help='Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.')

    parser.add_argument('--slots_fuse_asr_decoder_with_bert', action='store_true')

    return parser


def main(args):
    """Run the main decoding function."""
    parser = get_parser()
    args = parser.parse_args(args)

    if args.ngpu == 0 and args.dtype == "float16":
        raise ValueError(f"--dtype {args.dtype} does not support the CPU backend.")

    # logging info
    if args.verbose == 1:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose == 2:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # check CUDA_VISIBLE_DEVICES
    if args.ngpu > 0:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is None:
            logging.warning("CUDA_VISIBLE_DEVICES is not set.")
        elif args.ngpu != len(cvd.split(",")):
            logging.error("#gpus is not matched with CUDA_VISIBLE_DEVICES.")
            sys.exit(1)

        # TODO(mn5k): support of multiple GPUs
        if args.ngpu > 1:
            logging.error("The program only supports ngpu=1.")
            sys.exit(1)

    # display PYTHONPATH
    logging.info("python path = " + os.environ.get("PYTHONPATH", "(None)"))

    # seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    logging.info("set random seed = %d" % args.seed)

    # validate rnn options
    if args.rnnlm is not None and args.word_rnnlm is not None:
        logging.error(
            "It seems that both --rnnlm and --word-rnnlm are specified. "
            "Please use either option."
        )
        sys.exit(1)

    # slots related
    #assert args.slots_testing == bool(args.slots_out_path) == bool(args.slots_read_model) ==  bool(args.slots_read_vocab)

    #if args.slots_test_batchSize == 0:
    #    args.slots_test_batchSize = args.slots_batchSize
    #assert args.slots_batchSize % args.slots_gradient_accumulation_steps == 0

    assert args.slots_task_st in {'NN', 'NN_crf'}
    assert args.slots_task_sc in {'none', 'CLS', 'max', 'CLS_max', 'CLS_decoder'}
    assert args.slots_sc_type in {'single_cls_CE', 'multi_cls_BCE'}

    if args.slots_sc_type == 'multi_cls_BCE':
        args.slots_multiClass = True
    else:
        args.slots_multiClass = False
    if args.slots_task_st == 'NN_crf':
        args.slots_crf = True
    else:
        args.slots_crf = False

    assert 0 < args.slots_st_weight <= 1
    if args.slots_st_weight == 1 or args.slots_task_sc == 'none':
        args.slots_task_sc = None

    # recog
    logging.info("backend = " + args.backend)
    if args.num_spkrs == 1:
        if args.backend == "chainer":
            from espnet.asr.chainer_backend.asr import recog

            recog(args)
        elif args.backend == "pytorch":
            if args.num_encs == 1:
                # Experimental API that supports custom LMs
                if args.api == "v2":
                    from espnet.asr.pytorch_backend.recog import recog_v2

                    recog_v2(args)
                else:
                    from espnet.asr.pytorch_backend.asr_bert import recog

                    if args.dtype != "float32":
                        raise NotImplementedError(
                            f"`--dtype {args.dtype}` is only available with `--api v2`"
                        )
                    recog(args)
            else:
                if args.api == "v2":
                    raise NotImplementedError(
                        f"--num-encs {args.num_encs} > 1 is not supported in --api v2"
                    )
                else:
                    from espnet.asr.pytorch_backend.asr import recog

                    recog(args)
        else:
            raise ValueError("Only chainer and pytorch are supported.")
    elif args.num_spkrs == 2:
        if args.backend == "pytorch":
            from espnet.asr.pytorch_backend.asr_mix import recog

            recog(args)
        else:
            raise ValueError("Only pytorch is supported.")


if __name__ == "__main__":
    main(sys.argv[1:])
