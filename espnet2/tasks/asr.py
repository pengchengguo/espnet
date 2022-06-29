import argparse
import logging
from typing import Callable
from typing import Collection
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from typeguard import check_argument_types
from typeguard import check_return_type

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.decoder.mlm_decoder import MLMDecoder
from espnet2.asr.decoder.rnn_decoder import RNNDecoder
from espnet2.asr.decoder.transformer_decoder import (
    DynamicConvolution2DTransformerDecoder,  # noqa: H301
)
from espnet2.asr.decoder.transformer_decoder import DynamicConvolutionTransformerDecoder
from espnet2.asr.decoder.transformer_decoder import (
    LightweightConvolution2DTransformerDecoder,  # noqa: H301
)
from espnet2.asr.decoder.transformer_decoder import (
    LightweightConvolutionTransformerDecoder,  # noqa: H301
)
from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
from espnet2.asr.encoder.longformer_encoder import LongformerEncoder

from espnet2.asr.encoder.hubert_encoder import FairseqHubertEncoder
from espnet2.asr.encoder.hubert_encoder import FairseqHubertPretrainEncoder
from espnet2.asr.encoder.rnn_encoder import RNNEncoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.encoder.contextual_block_transformer_encoder import (
    ContextualBlockTransformerEncoder,  # noqa: H301
)
from espnet2.asr.encoder.contextual_block_conformer_encoder import (
    ContextualBlockConformerEncoder,  # noqa: H301
)
from espnet2.asr.encoder.vgg_rnn_encoder import VGGRNNEncoder
from espnet2.asr.encoder.wav2vec2_encoder import FairSeqWav2Vec2Encoder
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.asr.featurizer.featurizer import (
    WeightedSumFeaturizer,
    IdentityFeaturizer,
)
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.frontend.fused import FusedFrontends
from espnet2.asr.frontend.s3prl import S3prlFrontend
from espnet2.asr.frontend.windowing import SlidingWindow
from espnet2.asr.maskctc_model import MaskCTCModel
from espnet2.asr.hier_model import HierASRModel
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.postencoder.hugging_face_transformers_postencoder import (
    HuggingFaceTransformersPostEncoder,  # noqa: H301
)
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.preencoder.linear import LinearProjection
from espnet2.asr.preencoder.sinc import LightweightSincConvs
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.specaug.specaug import SpecAug
from espnet2.asr.transducer.joint_network import JointNetwork
from espnet2.asr.transducer.transducer_decoder import TransducerDecoder
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.tasks.abs_task import AbsTask
from espnet2.text.phoneme_tokenizer import g2p_choices
from espnet2.torch_utils.initialize import initialize
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.preprocessor import CommonPreprocessor
from espnet2.train.preprocessor import MutliTokenizerCommonPreprocessor
from espnet2.train.trainer import Trainer
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import float_or_none
from espnet2.utils.types import int_or_none
from espnet2.utils.types import str2bool
from espnet2.utils.types import str_or_none

frontend_choices = ClassChoices(
    name="frontend",
    classes=dict(
        default=DefaultFrontend,
        sliding_window=SlidingWindow,
        s3prl=S3prlFrontend,
        fused=FusedFrontends,
    ),
    type_check=AbsFrontend,
    default="default",
)
specaug_choices = ClassChoices(
    name="specaug",
    classes=dict(specaug=SpecAug,),
    type_check=AbsSpecAug,
    default=None,
    optional=True,
)
normalize_choices = ClassChoices(
    "normalize",
    classes=dict(global_mvn=GlobalMVN, utterance_mvn=UtteranceMVN,),
    type_check=AbsNormalize,
    default="utterance_mvn",
    optional=True,
)
model_choices = ClassChoices(
    "model",
    classes=dict(espnet=ESPnetASRModel, maskctc=MaskCTCModel, hier=HierASRModel),
    type_check=AbsESPnetModel,
    default="espnet",
)
preencoder_choices = ClassChoices(
    name="preencoder",
    classes=dict(sinc=LightweightSincConvs, linear=LinearProjection,),
    type_check=AbsPreEncoder,
    default=None,
    optional=True,
)
encoder_choices = ClassChoices(
    "encoder",
    classes=dict(
        conformer=ConformerEncoder,
        transformer=TransformerEncoder,
        contextual_block_transformer=ContextualBlockTransformerEncoder,
        contextual_block_conformer=ContextualBlockConformerEncoder,
        vgg_rnn=VGGRNNEncoder,
        rnn=RNNEncoder,
        wav2vec2=FairSeqWav2Vec2Encoder,
        hubert=FairseqHubertEncoder,
        hubert_pretrain=FairseqHubertPretrainEncoder,
        longformer=LongformerEncoder,
    ),
    type_check=AbsEncoder,
    default="rnn",
)
postencoder_choices = ClassChoices(
    name="postencoder",
    classes=dict(hugging_face_transformers=HuggingFaceTransformersPostEncoder,),
    type_check=AbsPostEncoder,
    default=None,
    optional=True,
)
featurizer_choices = ClassChoices(
    name="featurizer",
    classes=dict(weightedsum=WeightedSumFeaturizer, identity=IdentityFeaturizer),
    type_check=torch.nn.Module,
    default=None,
)
decoder_choices = ClassChoices(
    "decoder",
    classes=dict(
        transformer=TransformerDecoder,
        lightweight_conv=LightweightConvolutionTransformerDecoder,
        lightweight_conv2d=LightweightConvolution2DTransformerDecoder,
        dynamic_conv=DynamicConvolutionTransformerDecoder,
        dynamic_conv2d=DynamicConvolution2DTransformerDecoder,
        rnn=RNNDecoder,
        transducer=TransducerDecoder,
        mlm=MLMDecoder,
    ),
    type_check=AbsDecoder,
    default="rnn",
)


class ASRTask(AbsTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    # Add variable objects configurations
    class_choices_list = [
        # --frontend and --frontend_conf
        frontend_choices,
        # --specaug and --specaug_conf
        specaug_choices,
        # --normalize and --normalize_conf
        normalize_choices,
        # --model and --model_conf
        model_choices,
        # --preencoder and --preencoder_conf
        preencoder_choices,
        # --encoder and --encoder_conf
        encoder_choices,
        # --postencoder and --postencoder_conf
        postencoder_choices,
        # --featurizer and --featurizer_conf
        featurizer_choices,
        # --decoder and --decoder_conf
        decoder_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        required = parser.get_default("required")
        required += ["token_list"]

        group.add_argument(
            "--token_list",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to token",
        )
        group.add_argument(
            "--init",
            type=lambda x: str_or_none(x.lower()),
            default=None,
            help="The initialization method",
            choices=[
                "chainer",
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
                None,
            ],
        )

        group.add_argument(
            "--input_size",
            type=int_or_none,
            default=None,
            help="The number of input dimension of the feature",
        )

        group.add_argument(
            "--ctc_conf",
            action=NestedDictAction,
            default=get_default_kwargs(CTC),
            help="The keyword arguments for CTC class.",
        )
        group.add_argument(
            "--joint_net_conf",
            action=NestedDictAction,
            default=None,
            help="The keyword arguments for joint network class.",
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=True,
            help="Apply preprocessing to data or not",
        )
        group.add_argument(
            "--token_type",
            type=str,
            default="bpe",
            choices=["bpe", "char", "word", "phn"],
            help="The text will be tokenized " "in the specified level token",
        )
        group.add_argument(
            "--bpemodel",
            type=str_or_none,
            default=None,
            help="The model file of sentencepiece",
        )
        parser.add_argument(
            "--non_linguistic_symbols",
            type=str_or_none,
            help="non_linguistic_symbols file path",
        )
        parser.add_argument(
            "--cleaner",
            type=str_or_none,
            choices=[None, "tacotron", "jaconv", "vietnamese"],
            default=None,
            help="Apply text cleaning",
        )
        parser.add_argument(
            "--g2p",
            type=str_or_none,
            choices=g2p_choices,
            default=None,
            help="Specify g2p method if --token_type=phn",
        )
        parser.add_argument(
            "--speech_volume_normalize",
            type=float_or_none,
            default=None,
            help="Scale the maximum amplitude to the given value.",
        )
        parser.add_argument(
            "--rir_scp",
            type=str_or_none,
            default=None,
            help="The file path of rir scp file.",
        )
        parser.add_argument(
            "--rir_apply_prob",
            type=float,
            default=1.0,
            help="THe probability for applying RIR convolution.",
        )
        parser.add_argument(
            "--noise_scp",
            type=str_or_none,
            default=None,
            help="The file path of noise scp file.",
        )
        parser.add_argument(
            "--noise_apply_prob",
            type=float,
            default=1.0,
            help="The probability applying Noise adding.",
        )
        parser.add_argument(
            "--noise_db_range",
            type=str,
            default="13_15",
            help="The range of noise decibel level.",
        )

        # Multi-granular targets training
        parser.add_argument(
            "--bpemodels", type=str, action="append", default=[],
        )
        parser.add_argument(
            "--token_types",
            type=str,
            action="append",
            choices=["bpe", "char", "word", "phn"],
            default=[],
        )
        parser.add_argument("--token_lists", type=str, action="append", default=[])

        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    def build_collate_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        assert check_argument_types()
        # NOTE(kamo): int value = 0 is reserved by CTC-blank symbol
        return CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)

    @classmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        assert check_argument_types()
        if args.use_preprocessor:
            if len(args.token_types) == 0:
                logging.info("Single target training.")
                retval = CommonPreprocessor(
                    train=train,
                    token_type=args.token_type,
                    token_list=args.token_list,
                    bpemodel=args.bpemodel,
                    non_linguistic_symbols=args.non_linguistic_symbols,
                    text_cleaner=args.cleaner,
                    g2p_type=args.g2p,
                    # NOTE(kamo): Check attribute existence for backward compatibility
                    rir_scp=args.rir_scp if hasattr(args, "rir_scp") else None,
                    rir_apply_prob=args.rir_apply_prob
                    if hasattr(args, "rir_apply_prob")
                    else 1.0,
                    noise_scp=args.noise_scp if hasattr(args, "noise_scp") else None,
                    noise_apply_prob=args.noise_apply_prob
                    if hasattr(args, "noise_apply_prob")
                    else 1.0,
                    noise_db_range=args.noise_db_range
                    if hasattr(args, "noise_db_range")
                    else "13_15",
                    speech_volume_normalize=args.speech_volume_normalize
                    if hasattr(args, "rir_scp")
                    else None,
                )
            else:
                logging.info("Multi-granular targets training.")
                assert (
                    len(args.bpemodels)
                    == len(args.token_types)
                    == len(args.token_lists)
                )
                text_name = []
                for i, n in enumerate(args.bpemodels):
                    name = "char" if n == "none" else n.split("/")[-2]
                    text_name.append("text{}_".format(i) + name)

                retval = MutliTokenizerCommonPreprocessor(
                    train=train,
                    token_type=args.token_types,  # a list of token_types
                    token_list=args.token_lists,  # a list of token_lists
                    bpemodel=args.bpemodels,  # a list of BPE models
                    non_linguistic_symbols=args.non_linguistic_symbols,
                    text_cleaner=args.cleaner,
                    g2p_type=args.g2p,
                    # NOTE(kamo): Check attribute existence for backward compatibility
                    rir_scp=args.rir_scp if hasattr(args, "rir_scp") else None,
                    rir_apply_prob=args.rir_apply_prob
                    if hasattr(args, "rir_apply_prob")
                    else 1.0,
                    noise_scp=args.noise_scp if hasattr(args, "noise_scp") else None,
                    noise_apply_prob=args.noise_apply_prob
                    if hasattr(args, "noise_apply_prob")
                    else 1.0,
                    noise_db_range=args.noise_db_range
                    if hasattr(args, "noise_db_range")
                    else "13_15",
                    speech_volume_normalize=args.speech_volume_normalize
                    if hasattr(args, "rir_scp")
                    else None,
                    text_name=text_name,
                )
        else:
            retval = None
        assert check_return_type(retval)
        return retval

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        if not inference:
            retval = ("speech", "text")
        else:
            # Recognition mode
            retval = ("speech",)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        retval = ()
        assert check_return_type(retval)
        return retval

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> AbsESPnetModel:
        assert check_argument_types()
        if len(args.token_lists) == 0:
            # for single target training
            if isinstance(args.token_list, str):
                with open(args.token_list, encoding="utf-8") as f:
                    token_list = [line.rstrip() for line in f]

                # Overwriting token_list to keep it as "portable".
                args.token_list = list(token_list)
            elif isinstance(args.token_list, (tuple, list)):
                token_list = list(args.token_list)
            else:
                raise RuntimeError("token_list must be str or list")
            vocab_size = len(token_list)
            logging.info(f"Single target vocabulary size: {vocab_size }")
        else:
            # for mulit-granular targets training
            if isinstance(args.token_lists[0], str):
                token_lists = []
                for list_path in args.token_lists:
                    with open(list_path, encoding="utf-8") as f:
                        tl = [line.rstrip() for line in f]
                    token_lists.append(tl)

                # Overwriting token_lists to keep it as "portable".
                args.token_lists = list(token_lists)
            else:
                token_lists = list(args.token_lists)
            vocab_sizes = [len(tl) for tl in token_lists]
            logging.info(f"Multi-granular target vocabulary sizes: {vocab_sizes }")

        # 1. frontend
        if args.input_size is None:
            # Extract features in the model
            frontend_class = frontend_choices.get_class(args.frontend)
            frontend = frontend_class(**args.frontend_conf)
            input_size = frontend.output_size()
        else:
            # Give features from data-loader
            args.frontend = None
            args.frontend_conf = {}
            frontend = None
            input_size = args.input_size

        # 2. Data augmentation for spectrogram
        if args.specaug is not None:
            specaug_class = specaug_choices.get_class(args.specaug)
            specaug = specaug_class(**args.specaug_conf)
        else:
            specaug = None

        # 3. Normalization layer
        if args.normalize is not None:
            normalize_class = normalize_choices.get_class(args.normalize)
            normalize = normalize_class(**args.normalize_conf)
        else:
            normalize = None

        # 4. Pre-encoder input block
        # NOTE(kan-bayashi): Use getattr to keep the compatibility
        if getattr(args, "preencoder", None) is not None:
            preencoder_class = preencoder_choices.get_class(args.preencoder)
            preencoder = preencoder_class(**args.preencoder_conf)
            input_size = preencoder.output_size()
        else:
            preencoder = None

        # 4. Encoder
        encoder_class = encoder_choices.get_class(args.encoder)
        encoder = encoder_class(input_size=input_size, **args.encoder_conf)

        # 5. Post-encoder block
        # NOTE(kan-bayashi): Use getattr to keep the compatibility
        encoder_output_size = encoder.output_size()
        if getattr(args, "postencoder", None) is not None:
            postencoder_class = postencoder_choices.get_class(args.postencoder)
            postencoder = postencoder_class(
                input_size=encoder_output_size, **args.postencoder_conf
            )
            encoder_output_size = postencoder.output_size()
        else:
            postencoder = None

        # 5. Featurizer
        if getattr(args, "featurizer", None) is not None:
            featurizer_class = featurizer_choices.get_class(args.featurizer)
            featurizer = featurizer_class(**args.featurizer_conf)
        else:
            featurizer = None

        # 6. Decoder
        decoder_class = decoder_choices.get_class(args.decoder)

        if args.decoder == "transducer":
            assert len(args.token_lists) == 0, args.token_list
            decoder = decoder_class(vocab_size, embed_pad=0, **args.decoder_conf,)

            joint_network = JointNetwork(
                vocab_size,
                encoder.output_size(),
                decoder.dunits,
                **args.joint_net_conf,
            )
        else:
            if len(args.token_lists) == 0:
                decoder = decoder_class(
                    vocab_size=vocab_size,
                    encoder_output_size=encoder_output_size,
                    **args.decoder_conf,
                )
                logging.info(f"Decoder output size: {vocab_size }")
            else:
                decoder = decoder_class(
                    vocab_size=vocab_sizes[-1],
                    encoder_output_size=encoder_output_size,
                    **args.decoder_conf,
                )
                logging.info(f"Decoder output size: {vocab_sizes[-1] }")

            joint_network = None

        # 7. CTC
        if len(args.token_lists) == 0:
            ctc = CTC(
                odim=vocab_size,
                encoder_output_size=encoder_output_size,
                **args.ctc_conf,
            )
        else:
            ctc = torch.nn.ModuleList()
            num_interctc_layer = len(args.encoder_conf["interctc_layer_idx"])
            assert num_interctc_layer + 1 == len(vocab_sizes), num_interctc_layer
            for i in range(num_interctc_layer + 1):
                ctc.append(
                    CTC(
                        odim=vocab_sizes[i],
                        encoder_output_size=encoder_output_size,
                        **args.ctc_conf,
                    )
                )
                logging.info("CTC_{} output size: {}".format(i, vocab_sizes[i]))

        # 7. Build model
        try:
            model_class = model_choices.get_class(args.model)
        except AttributeError:
            model_class = model_choices.get_class("espnet")
        model = model_class(
            vocab_size=vocab_size if len(args.token_lists) == 0 else vocab_sizes,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            preencoder=preencoder,
            encoder=encoder,
            postencoder=postencoder,
            featurizer=featurizer,
            decoder=decoder,
            ctc=ctc,
            joint_network=joint_network,
            token_list=token_list if len(args.token_lists) == 0 else token_lists,
            **args.model_conf,
        )

        # FIXME(kamo): Should be done in model?
        # 8. Initialize
        if args.init is not None:
            initialize(model, args.init)

        assert check_return_type(model)
        return model
