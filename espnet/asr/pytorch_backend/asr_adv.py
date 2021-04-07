# Copyright 2021 Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Training/decoding definition for the adversarial speech recognition task."""

import json
import logging
import math
import os

from chainer import training
from chainer.training import extensions
from chainer.training.updater import StandardUpdater
from tensorboardX import SummaryWriter
import torch

from espnet.asr.asr_utils import adadelta_eps_decay
from espnet.asr.asr_utils import CompareValueTrigger
from espnet.asr.asr_utils import format_mulenc_args
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import restore_snapshot
from espnet.asr.asr_utils import snapshot_object
from espnet.asr.asr_utils import torch_load
from espnet.asr.asr_utils import torch_resume
from espnet.asr.asr_utils import torch_snapshot
from espnet.asr.pytorch_backend.asr import _recursive_to
from espnet.asr.pytorch_backend.asr import CustomConverter
from espnet.asr.pytorch_backend.asr import CustomConverterMulEnc
from espnet.asr.pytorch_backend.asr import CustomEvaluator
from espnet.asr.pytorch_backend.asr_init import freeze_modules
from espnet.asr.pytorch_backend.asr_init import load_trained_modules
from espnet.nets.asr_interface import ASRInterface
import espnet.nets.pytorch_backend.lm.default as lm_pytorch
from espnet.utils.dataset import ChainerDataLoader
from espnet.utils.dataset import TransformDataset
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.dynamic_import import dynamic_import
from espnet.utils.io_utils import LoadInputsAndTargets
from espnet.utils.training.batchfy import make_batchset
from espnet.utils.training.iterators import ShufflingEnabler
from espnet.utils.training.tensorboard_logger import TensorboardLogger
from espnet.utils.training.train_utils import check_early_stop
from espnet.utils.training.train_utils import set_early_stop

import matplotlib

matplotlib.use("Agg")


class CustomUpdater(StandardUpdater):
    """Custom Updater for Pytorch.

    Args:
        model (torch.nn.Module): The model to update.
        adversary (espnet.nets.pytorch_backend.adversary.*): The adversary class.
        grad_clip_threshold (float): The gradient clipping value to use.
        train_iter (chainer.dataset.Iterator): The training iterator.
        optimizer (torch.optim.optimizer): The training optimizer.

        device (torch.device): The device to use.
        ngpu (int): The number of gpus to use.
        use_apex (bool): The flag to use Apex in backprop.

    """

    def __init__(
        self,
        model,
        adversary,
        grad_clip_threshold,
        train_iter,
        optimizer,
        device,
        ngpu,
        grad_noise=False,
        accum_grad=1,
        use_apex=False,
    ):
        super(CustomUpdater, self).__init__(train_iter, optimizer)
        self.model = model
        self.adversary = adversary
        self.grad_clip_threshold = grad_clip_threshold
        self.device = device
        self.ngpu = ngpu
        self.accum_grad = accum_grad
        self.forward_count = 0
        self.grad_noise = grad_noise
        self.iteration = 0
        self.use_apex = use_apex

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        """Main update routine of the CustomUpdater."""
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator("main")
        optimizer = self.get_optimizer("main")
        epoch = train_iter.epoch

        # Get the next batch (a list of json files)
        batch = train_iter.next()
        # self.iteration += 1 # Increase may result in early report,
        # which is done in other place automatically.
        x = _recursive_to(batch, self.device)
        is_new_epoch = train_iter.epoch != epoch
        # When the last minibatch in the current epoch is given,
        # gradient accumulation is turned off in order to evaluate the model
        # on the validation set in every epoch.
        # see details in https://github.com/espnet/espnet/pull/1388

        # Compute the loss at this time step and accumulate it
        self.adversary(*x, epoch, self.accum_grad)

        # gradient noise injection
        if self.grad_noise:
            from espnet.asr.asr_utils import add_gradient_noise

            add_gradient_noise(
                self.model, self.iteration, duration=100, eta=1.0, scale_factor=0.55
            )

        # update parameters
        self.forward_count += 1
        if not is_new_epoch and self.forward_count != self.accum_grad:
            return
        self.forward_count = 0
        # compute the gradient norm to check if it is normal or not
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.grad_clip_threshold
        )
        logging.info("grad norm={}".format(grad_norm))
        if math.isnan(grad_norm):
            logging.warning("grad norm is nan. Do not update model.")
        else:
            optimizer.step()
        optimizer.zero_grad()

    def update(self):
        self.update_core()
        # #iterations with accum_grad > 1
        # Ref.: https://github.com/espnet/espnet/issues/777
        if self.forward_count == 0:
            self.iteration += 1


def train(args):
    """Train with the given args.

    Args:
        args (namespace): The program arguments.

    """
    set_deterministic_pytorch(args)
    if args.num_encs > 1:
        args = format_mulenc_args(args)

    # check cuda availability
    if not torch.cuda.is_available():
        logging.warning("cuda is not available")

    # get input and output dimension info
    with open(args.valid_json, "rb") as f:
        valid_json = json.load(f)["utts"]
    utts = list(valid_json.keys())
    idim_list = [
        int(valid_json[utts[0]]["input"][i]["shape"][-1]) for i in range(args.num_encs)
    ]
    odim = int(valid_json[utts[0]]["output"][0]["shape"][-1])
    for i in range(args.num_encs):
        logging.info("stream{}: input dims : {}".format(i + 1, idim_list[i]))
    logging.info("#output dims: " + str(odim))

    # specify attention, CTC, hybrid mode
    if "transducer" in args.model_module:
        if (
            getattr(args, "etype", False) == "transformer"
            or getattr(args, "dtype", False) == "transformer"
        ):
            mtl_mode = "transformer_transducer"
        else:
            mtl_mode = "transducer"
        logging.info("Pure transducer mode")
    elif args.mtlalpha == 1.0:
        mtl_mode = "ctc"
        logging.info("Pure CTC mode")
    elif args.mtlalpha == 0.0:
        mtl_mode = "att"
        logging.info("Pure attention mode")
    else:
        mtl_mode = "mtl"
        logging.info("Multitask learning mode")

    if (args.enc_init is not None or args.dec_init is not None) and args.num_encs == 1:
        model = load_trained_modules(idim_list[0], odim, args)
    else:
        model_class = dynamic_import(args.model_module)
        model = model_class(
            idim_list[0] if args.num_encs == 1 else idim_list, odim, args
        )
    assert isinstance(model, ASRInterface)
    total_subsampling_factor = model.get_total_subsampling_factor()

    logging.info(
        " Total parameter of the model = "
        + str(sum(p.numel() for p in model.parameters()))
    )

    if args.rnnlm is not None:
        rnnlm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(len(args.char_list), rnnlm_args.layer, rnnlm_args.unit)
        )
        torch_load(args.rnnlm, rnnlm)
        model.rnnlm = rnnlm

    # write model config
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    model_conf = args.outdir + "/model.json"
    with open(model_conf, "wb") as f:
        logging.info("writing a model config file to " + model_conf)
        f.write(
            json.dumps(
                (idim_list[0] if args.num_encs == 1 else idim_list, odim, vars(args)),
                indent=4,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf_8")
        )
    for key in sorted(vars(args).keys()):
        logging.info("ARGS: " + key + ": " + str(vars(args)[key]))

    reporter = model.reporter

    # check the use of multi-gpu
    if args.ngpu > 1:
        if args.batch_size != 0:
            logging.warning(
                "batch size is automatically increased (%d -> %d)"
                % (args.batch_size, args.batch_size * args.ngpu)
            )
            args.batch_size *= args.ngpu
        if args.num_encs > 1:
            # TODO(ruizhili): implement data parallel for multi-encoder setup.
            raise NotImplementedError(
                "Data parallel is not supported for multi-encoder setup."
            )

    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    if args.train_dtype in ("float16", "float32", "float64"):
        dtype = getattr(torch, args.train_dtype)
    else:
        dtype = torch.float32
    model = model.to(device=device, dtype=dtype)

    if args.freeze_mods:
        model, model_params = freeze_modules(model, args.freeze_mods)
    else:
        model_params = model.parameters()

    logging.warning(
        "num. model params: {:,} (num. trained: {:,} ({:.1f}%))".format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
            sum(p.numel() for p in model.parameters() if p.requires_grad)
            * 100.0
            / sum(p.numel() for p in model.parameters()),
        )
    )

    # Setup an optimizer
    if args.opt == "adadelta":
        optimizer = torch.optim.Adadelta(
            model_params, rho=0.95, eps=args.eps, weight_decay=args.weight_decay
        )
    elif args.opt == "adam":
        optimizer = torch.optim.Adam(model_params, weight_decay=args.weight_decay)
    elif args.opt == "noam":
        from espnet.nets.pytorch_backend.transformer.optimizer import get_std_opt

        # For transformer-transducer, adim declaration is within the block definition.
        # Thus, we need retrieve the most dominant value (d_hidden) for Noam scheduler.
        if hasattr(args, "enc_block_arch") or hasattr(args, "dec_block_arch"):
            adim = model.most_dom_dim
        else:
            adim = args.adim

        optimizer = get_std_opt(
            model_params, adim, args.transformer_warmup_steps, args.transformer_lr
        )
    else:
        raise NotImplementedError("unknown optimizer: " + args.opt)

    # setup apex.amp
    if args.train_dtype in ("O0", "O1", "O2", "O3"):
        try:
            from apex import amp
        except ImportError as e:
            logging.error(
                f"You need to install apex for --train-dtype {args.train_dtype}. "
                "See https://github.com/NVIDIA/apex#linux"
            )
            raise e
        if args.opt == "noam":
            model, optimizer.optimizer = amp.initialize(
                model, optimizer.optimizer, opt_level=args.train_dtype
            )
        else:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level=args.train_dtype
            )
        use_apex = True

        from espnet.nets.pytorch_backend.ctc import CTC

        amp.register_float_function(CTC, "loss_fn")
        amp.init()
        logging.warning("register ctc as float function")
    else:
        use_apex = False

    # FIXME: TOO DIRTY HACK
    setattr(optimizer, "target", reporter)
    setattr(optimizer, "serialize", lambda s: reporter.serialize(s))

    # Setup a converter
    if args.num_encs == 1:
        converter = CustomConverter(subsampling_factor=model.subsample[0], dtype=dtype)
    else:
        converter = CustomConverterMulEnc(
            [i[0] for i in model.subsample_list], dtype=dtype
        )

    # read json data
    with open(args.train_json, "rb") as f:
        train_json = json.load(f)["utts"]
    with open(args.valid_json, "rb") as f:
        valid_json = json.load(f)["utts"]

    use_sortagrad = args.sortagrad == -1 or args.sortagrad > 0
    # make minibatch list (variable length)
    train = make_batchset(
        train_json,
        args.batch_size,
        args.maxlen_in,
        args.maxlen_out,
        args.minibatches,
        min_batch_size=args.ngpu if args.ngpu > 1 else 1,
        shortest_first=use_sortagrad,
        count=args.batch_count,
        batch_bins=args.batch_bins,
        batch_frames_in=args.batch_frames_in,
        batch_frames_out=args.batch_frames_out,
        batch_frames_inout=args.batch_frames_inout,
        iaxis=0,
        oaxis=0,
    )
    valid = make_batchset(
        valid_json,
        args.batch_size,
        args.maxlen_in,
        args.maxlen_out,
        args.minibatches,
        min_batch_size=args.ngpu if args.ngpu > 1 else 1,
        count=args.batch_count,
        batch_bins=args.batch_bins,
        batch_frames_in=args.batch_frames_in,
        batch_frames_out=args.batch_frames_out,
        batch_frames_inout=args.batch_frames_inout,
        iaxis=0,
        oaxis=0,
    )

    load_tr = LoadInputsAndTargets(
        mode="asr",
        load_output=True,
        preprocess_conf=args.preprocess_conf,
        preprocess_args={"train": True},  # Switch the mode of preprocessing
    )
    load_cv = LoadInputsAndTargets(
        mode="asr",
        load_output=True,
        preprocess_conf=args.preprocess_conf,
        preprocess_args={"train": False},  # Switch the mode of preprocessing
    )
    # hack to make batchsize argument as 1
    # actual bathsize is included in a list
    # default collate function converts numpy array to pytorch tensor
    # we used an empty collate function instead which returns list
    train_iter = ChainerDataLoader(
        dataset=TransformDataset(train, lambda data: converter([load_tr(data)])),
        batch_size=1,
        num_workers=args.n_iter_processes,
        shuffle=not use_sortagrad,
        collate_fn=lambda x: x[0],
    )
    valid_iter = ChainerDataLoader(
        dataset=TransformDataset(valid, lambda data: converter([load_cv(data)])),
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x[0],
        num_workers=args.n_iter_processes,
    )

    # Set up an adversarial for adversarial training
    if args.adv_type == "freelb":
        from espnet.nets.pytorch_backend.adversary.freelb import FreeLB

        adversary = FreeLB(
            model,
            adv_prob=args.adv_prob,
            adv_lr=args.adv_lr,
            adv_steps=args.adv_steps,
            rand_init_mag=args.rand_init_mag,
            adv_begin_epoch=args.adv_begin_epoch,
            no_sync_dp=args.no_sync_dp,
            max_norm=args.max_norm,
            norm_method=args.norm_method,
            reg_weight=args.reg_weight,
        )

        # modify the epoch based on adv_steps
        # logging.warning(
        #     "For FreeLB training, training epochs are automatically decreased (%d -> %d)."
        #     % (args.epochs, args.epochs // args.adv_steps + 1)
        # )
        # args.epochs = args.epochs // args.adv_steps + 1

    else:
        raise ValueError("Not implemented.")

    # Set up a trainer
    updater = CustomUpdater(
        model,
        adversary,
        args.grad_clip,
        {"main": train_iter},
        optimizer,
        device,
        args.ngpu,
        args.grad_noise,
        args.accum_grad,
        use_apex=use_apex,
    )
    trainer = training.Trainer(updater, (args.epochs, "epoch"), out=args.outdir)

    if use_sortagrad:
        trainer.extend(
            ShufflingEnabler([train_iter]),
            trigger=(args.sortagrad if args.sortagrad != -1 else args.epochs, "epoch"),
        )

    # Resume from a snapshot
    if args.resume:
        logging.info("resumed from %s" % args.resume)
        torch_resume(args.resume, trainer)

    # Evaluate the model with the test dataset for each epoch
    if args.save_interval_iters > 0:
        trainer.extend(
            CustomEvaluator(model, {"main": valid_iter}, reporter, device, args.ngpu),
            trigger=(args.save_interval_iters, "iteration"),
        )
    else:
        trainer.extend(
            CustomEvaluator(model, {"main": valid_iter}, reporter, device, args.ngpu)
        )

    # Save attention weight each epoch
    is_attn_plot = (
        "transformer" in args.model_module
        or "conformer" in args.model_module
        or mtl_mode in ["att", "mtl"]
    ) or mtl_mode == "transformer_transducer"

    if args.num_save_attention > 0 and is_attn_plot:
        data = sorted(
            list(valid_json.items())[: args.num_save_attention],
            key=lambda x: int(x[1]["input"][0]["shape"][1]),
            reverse=True,
        )
        if hasattr(model, "module"):
            att_vis_fn = model.module.calculate_all_attentions
            plot_class = model.module.attention_plot_class
        else:
            att_vis_fn = model.calculate_all_attentions
            plot_class = model.attention_plot_class
        att_reporter = plot_class(
            att_vis_fn,
            data,
            args.outdir + "/att_ws",
            converter=converter,
            transform=load_cv,
            device=device,
            subsampling_factor=total_subsampling_factor,
        )
        trainer.extend(att_reporter, trigger=(1, "epoch"))
    else:
        att_reporter = None

    # Save CTC prob at each epoch
    if mtl_mode in ["ctc", "mtl"] and args.num_save_ctc > 0:
        # NOTE: sort it by output lengths
        data = sorted(
            list(valid_json.items())[: args.num_save_ctc],
            key=lambda x: int(x[1]["output"][0]["shape"][0]),
            reverse=True,
        )
        if hasattr(model, "module"):
            ctc_vis_fn = model.module.calculate_all_ctc_probs
            plot_class = model.module.ctc_plot_class
        else:
            ctc_vis_fn = model.calculate_all_ctc_probs
            plot_class = model.ctc_plot_class
        ctc_reporter = plot_class(
            ctc_vis_fn,
            data,
            args.outdir + "/ctc_prob",
            converter=converter,
            transform=load_cv,
            device=device,
            subsampling_factor=total_subsampling_factor,
        )
        trainer.extend(ctc_reporter, trigger=(1, "epoch"))
    else:
        ctc_reporter = None

    # Make a plot for training and validation values
    if args.num_encs > 1:
        report_keys_loss_ctc = [
            "main/loss_ctc{}".format(i + 1) for i in range(model.num_encs)
        ] + ["validation/main/loss_ctc{}".format(i + 1) for i in range(model.num_encs)]
        report_keys_cer_ctc = [
            "main/cer_ctc{}".format(i + 1) for i in range(model.num_encs)
        ] + ["validation/main/cer_ctc{}".format(i + 1) for i in range(model.num_encs)]
    trainer.extend(
        extensions.PlotReport(
            [
                "main/loss",
                "validation/main/loss",
                "main/loss_ctc",
                "validation/main/loss_ctc",
                "main/loss_att",
                "validation/main/loss_att",
            ]
            + ([] if args.num_encs == 1 else report_keys_loss_ctc),
            "epoch",
            file_name="loss.png",
        )
    )
    trainer.extend(
        extensions.PlotReport(
            ["main/acc", "validation/main/acc"], "epoch", file_name="acc.png"
        )
    )
    trainer.extend(
        extensions.PlotReport(
            ["main/cer_ctc", "validation/main/cer_ctc"]
            + ([] if args.num_encs == 1 else report_keys_loss_ctc),
            "epoch",
            file_name="cer.png",
        )
    )

    # Save best models
    trainer.extend(
        snapshot_object(model, "model.loss.best"),
        trigger=training.triggers.MinValueTrigger("validation/main/loss"),
    )
    if mtl_mode not in ["ctc", "transducer", "transformer_transducer"]:
        trainer.extend(
            snapshot_object(model, "model.acc.best"),
            trigger=training.triggers.MaxValueTrigger("validation/main/acc"),
        )

    # save snapshot which contains model and optimizer states
    if args.save_interval_iters > 0:
        trainer.extend(
            torch_snapshot(filename="snapshot.iter.{.updater.iteration}"),
            trigger=(args.save_interval_iters, "iteration"),
        )

    # save snapshot at every epoch - for model averaging
    trainer.extend(torch_snapshot(), trigger=(1, "epoch"))

    # epsilon decay in the optimizer
    if args.opt == "adadelta":
        if args.criterion == "acc" and mtl_mode != "ctc":
            trainer.extend(
                restore_snapshot(
                    model, args.outdir + "/model.acc.best", load_fn=torch_load
                ),
                trigger=CompareValueTrigger(
                    "validation/main/acc",
                    lambda best_value, current_value: best_value > current_value,
                ),
            )
            trainer.extend(
                adadelta_eps_decay(args.eps_decay),
                trigger=CompareValueTrigger(
                    "validation/main/acc",
                    lambda best_value, current_value: best_value > current_value,
                ),
            )
        elif args.criterion == "loss":
            trainer.extend(
                restore_snapshot(
                    model, args.outdir + "/model.loss.best", load_fn=torch_load
                ),
                trigger=CompareValueTrigger(
                    "validation/main/loss",
                    lambda best_value, current_value: best_value < current_value,
                ),
            )
            trainer.extend(
                adadelta_eps_decay(args.eps_decay),
                trigger=CompareValueTrigger(
                    "validation/main/loss",
                    lambda best_value, current_value: best_value < current_value,
                ),
            )
        # NOTE: In some cases, it may take more than one epoch for the model's loss
        # to escape from a local minimum.
        # Thus, restore_snapshot extension is not used here.
        # see details in https://github.com/espnet/espnet/pull/2171
        elif args.criterion == "loss_eps_decay_only":
            trainer.extend(
                adadelta_eps_decay(args.eps_decay),
                trigger=CompareValueTrigger(
                    "validation/main/loss",
                    lambda best_value, current_value: best_value < current_value,
                ),
            )

    # Write a log of evaluation statistics for each epoch
    trainer.extend(
        extensions.LogReport(trigger=(args.report_interval_iters, "iteration"))
    )
    report_keys = [
        "epoch",
        "iteration",
        "main/loss",
        "main/loss_ctc",
        "main/loss_att",
        "validation/main/loss",
        "validation/main/loss_ctc",
        "validation/main/loss_att",
        "main/acc",
        "validation/main/acc",
        "main/cer_ctc",
        "validation/main/cer_ctc",
        "elapsed_time",
    ] + ([] if args.num_encs == 1 else report_keys_cer_ctc + report_keys_loss_ctc)
    if args.opt == "adadelta":
        trainer.extend(
            extensions.observe_value(
                "eps",
                lambda trainer: trainer.updater.get_optimizer("main").param_groups[0][
                    "eps"
                ],
            ),
            trigger=(args.report_interval_iters, "iteration"),
        )
        report_keys.append("eps")
    if args.report_cer:
        report_keys.append("validation/main/cer")
    if args.report_wer:
        report_keys.append("validation/main/wer")
    trainer.extend(
        extensions.PrintReport(report_keys),
        trigger=(args.report_interval_iters, "iteration"),
    )

    trainer.extend(extensions.ProgressBar(update_interval=args.report_interval_iters))
    set_early_stop(trainer, args)

    if args.tensorboard_dir is not None and args.tensorboard_dir != "":
        trainer.extend(
            TensorboardLogger(
                SummaryWriter(args.tensorboard_dir),
                att_reporter=att_reporter,
                ctc_reporter=ctc_reporter,
            ),
            trigger=(args.report_interval_iters, "iteration"),
        )
    # Run the training
    trainer.run()
    check_early_stop(trainer, args.epochs)
