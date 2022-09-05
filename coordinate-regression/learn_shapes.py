from argparse import ArgumentParser
from itertools import chain
from types import SimpleNamespace
from typing import Optional

import torch
import norse.torch as norse
import pytorch_lightning as pl

import shape_dataset
from .model import *
from .loss import *


class ShapesModel(pl.LightningModule):
    def __init__(
        self,
        net: str,
        single_class: Optional[int],
        li_tau_syn_inv: float,
        li_tau_mem_inv: float,
        lif_tau_syn_inv: float,
        lif_tau_mem_inv: float,
        method: str,
        v_leak: float,
        v_th: float,
        coordinate: str,
        rectification: str,
        lr: float,
        lr_step: str,
        resolution: torch.Size,
        warmup: int,
        learn_parameters: bool,
        optimizer: str,
        device: str,
        **kwargs,
    ):
        super().__init__()

        # Network
        p_li = norse.LIParameters(
            tau_syn_inv=torch.as_tensor(li_tau_syn_inv, device=device),
            tau_mem_inv=torch.as_tensor(li_tau_mem_inv, device=device),
            v_leak=torch.as_tensor(v_leak, device=device),
        )
        p_lif = norse.LIFParameters(
            tau_syn_inv=torch.as_tensor(lif_tau_syn_inv, device=device),
            tau_mem_inv=torch.as_tensor(lif_tau_mem_inv, device=device),
            v_leak=torch.as_tensor(v_leak, device=device),
            v_th=torch.as_tensor(v_th, device=device),
            method=method,
        )
        classes = 1 if single_class is not None else 3
        kernels = (40, 20, 3)
        if net == "ann":
            self.net = ANN(kernels=kernels, classes=classes)
        elif net == "annrf":
            self.net = ANNRF(rings_file="rings_20.dat", classes=classes)
        elif net == "snn":
            self.net = ShapesSNNLayer(
                p_li,
                p_lif,
                kernels=kernels,
                classes=classes,
                learn_parameters=learn_parameters,
            )
        elif net == "snnrf":
            self.net = ShapesSNNRFLayer(p_li, p_lif, "rings_20.dat", classes=classes)
        
        else:
            raise ValueError("Unknown network type " + net)

        # Rectification
        if rectification == "softmax":
            self.rectification = torch.nn.Softmax(dim=-1)
        elif rectification == "sigmoid":
            self.rectification = torch.nn.Sigmoid()
        elif rectification == "relu":
            self.rectification = torch.nn.ReLU()
        else:
            self.rectification = torch.nn.Identity()
        
        # Coordinate
        if coordinate == "dsnt":
            self.coordinate = DSNT(self.net.out_shape)
        elif coordinate == "dsntli":
            self.coordinate = DSNTLI(self.net.out_shape)
        else:
            self.coordinate = PixelActivityToCoordinate(resolution)

        self.resolution = torch.tensor(resolution)
        self.lr = lr
        self.lr_step = lr_step
        self.warmup = warmup
        self.optimizer = optimizer
        self.single_class = single_class

        self.save_hyperparameters()
        self.save_hyperparameters(kwargs)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Network Model")
        parser.add_argument(
            "--net",
            type=str,
            default="ann",
            choices=["ann", "annrf", "snn", "snnrf"],
        )
        parser.add_argument(
            "--rectification",
            type=str,
            choices=["softmax", "relu", "id", "sigmoid"],
            default="softmax",
        )
        parser.add_argument(
            "--coordinate",
            type=str,
            choices=["dsnt", "dsntli", "pixel"],
            default="dsntli",
            help="Method to reduce 2d surface to coordinate",
        )
        parser.add_argument(
            "--learn_parameters",
            action="store_true",
            default=False,
            help="Optimize LI parameters?",
        )
        parser.add_argument("--lr", type=float, default=8e-4)
        parser.add_argument(
            "--lr_step", type=str, default="step", choices=["step", "ca", "none"]
        )
        parser.add_argument("--li_tau_syn_inv", type=float, default=400)
        parser.add_argument("--li_tau_mem_inv", type=float, default=800)
        parser.add_argument("--lif_tau_syn_inv", type=float, default=700)
        parser.add_argument("--lif_tau_mem_inv", type=float, default=900)
        parser.add_argument("--v_leak", type=float, default=0.0)
        parser.add_argument("--v_th", type=float, default=0.7)
        parser.add_argument(
            "--method",
            type=str,
            choices=["super", "triangle", "tanh", "adjoint"],
            default="super",
        )
        parser.add_argument(
            "--optimizer", type=str, choices=["adam", "rmsprop"], default="rmsprop"
        )
        return parent_parser

    def extract_kernels(self, net):
        ks = []
        for m in net.children():
            ks += self.extract_kernels(m)

        if isinstance(net, norse.Lift):
            ks += self.extract_kernels(net.lifted_module)
        elif isinstance(net, torch.nn.Conv2d) or isinstance(
            net, torch.nn.ConvTranspose2d
        ):
            ks.append(net.weight.clone().detach().cpu())
        return ks

    def normalized_to_image(self, coordinate):
        return ((coordinate + 1) * self.resolution.to(self.device)) * 0.5

    def image_to_normalized(self, coordinate):
        return ((coordinate * 2) / self.resolution.to(self.device)) - 1

    def extract_batch(self, batch):
        warmup, x, y_co = batch
        warmup = warmup.permute(1, 0, 2, 3, 4)  # TBCXY
        x = x.permute(1, 0, 2, 3, 4)  # TBCXY
        # Use events for imaging
        y_im = x
        y_co = y_co[:, :]
        # Restrict to single class, if requested
        if self.single_class is not None:
            y_co = y_co[:, :, self.single_class].unsqueeze(-2)
        y_co = y_co.permute(1, 0, 2, 3)  # TBCP
        y_co_norm = self.image_to_normalized(y_co)
        return warmup, x, y_co, y_co_norm, y_im

    def calc_coordinate(self, activations, state=None):
        rectified = self.rectification(activations.flatten(3)).reshape(
            activations.shape
        )
        coordinate = self.coordinate(rectified, state)
        return rectified, coordinate

    def forward(self, warmup, x, prepend_warmup: bool = False):
        # Warmup
        with torch.no_grad():
            out_warmup, s = self.net(warmup)
            _, (co_warmup, co_s) = self.calc_coordinate(out_warmup)

        # Predict
        out, _ = self.net(x, s)
        out, (out_co, _) = self.calc_coordinate(out, co_s)  # Replace out w/ rectified
        if prepend_warmup:
            return torch.cat((out_warmup, out_co)), torch.cat((co_warmup, out_co))
        return out, out_co

    def calc_loss(self, out, out_co, y_co):
        # Norm
        return torch.norm(out_co - y_co, p=2, dim=-1)

    def training_step(self, batch, batch_idx):
        warmup, x, y_co, y_co_norm, y_im = self.extract_batch(batch)
        out, out_co = self.forward(warmup, x)
        loss_co = self.calc_loss(out, out_co, y_co_norm)
        loss = loss_co.mean()

        self.log("train/loss", loss.mean())
        if self.lr_schedulers() is not None:
            self.log("lr", self.lr_schedulers().get_last_lr()[0])

        return loss.mean()

    def validation_step(self, batch, batch_idx):
        warmup, x, y_co, y_co_norm, y_im = self.extract_batch(batch)
        out, out_co = self.forward(warmup, x)
        loss_co = self.calc_loss(out, out_co, y_co_norm)
        loss = loss_co.mean() 

        # Log prediction
        # Randomize shape
        if self.single_class is not None:
            shape_id = 0
        else:
            shape_id = torch.randint(0, 3, (1,)).item()

        dic = {
            "loss": loss.mean(),
            "hp_metric": loss.mean(),
        }

        self.log("val/loss", loss.mean())
        self.log("hp_metric", loss.mean())
        return dic

    def predict_step(self, batch, batch_idx):
        w, x, l = batch
        warmup, x, y_co, y_co_norm, y_im = self.extract_batch(
            [w.unsqueeze(0), x.unsqueeze(0), l.unsqueeze(0)]
        )  # Add fake batch dim
        out, out_co = self.forward(warmup, x)
        return self.normalized_to_image(out_co), y_co

    def configure_optimizers(self):
        params = chain(
            self.net.parameters(),
            self.coordinate.parameters(),
            self.rectification.parameters(),
        )
        if self.optimizer == "adam":
            optim = torch.optim.Adam(params, lr=self.lr, weight_decay=1e-5)
        else:
            optim = torch.optim.RMSprop(params, lr=self.lr, weight_decay=1e-5)
        if self.lr_step == "step":
            stepper = [torch.optim.lr_scheduler.ExponentialLR(optim, 0.95)]
        elif self.lr_step == "ca":
            stepper = [
                torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=10)
            ]
        elif self.lr_step == "none":
            stepper = []
        else:
            raise ValueError("Unknown stepper")
        return [optim], stepper


def train(args, callbacks=[]):
    torch.manual_seed(0)
    args.resolution = (640, 480)

    if args.mode == "single_triangle_top":
        data_offset = torch.tensor([0, -40])
        args.single_class = 2  # Triangle
    if args.mode == "single_triangle":
        data_offset = torch.tensor([0, -40])
        args.single_class = 2  # Triangle
    elif args.mode == "single":
        data_offset = torch.zeros(2)
        args.single_class = 0  # Circle
    else:
        data_offset = torch.zeros(2)
        args.single_class = None

    train_data = torch.utils.data.DataLoader(
        shape_dataset.ShapeDataset(
            args.data_root,
            t=args.timesteps,
            pose_offset=data_offset,
            pose_delay=args.network_delay,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        prefetch_factor=4,
        drop_last=True,
    )
    val_data = torch.utils.data.DataLoader(
        shape_dataset.ShapeDataset(
            args.data_root,
            t=args.timesteps,
            train=False,
            pose_offset=data_offset,
            pose_delay=args.network_delay,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        prefetch_factor=3,
        drop_last=True,
    )
    name = f"{args.net}: {args.mode}"
    logger = pl.loggers.TensorBoardLogger("logs_shapes_nrp", name=name)

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=logger,
        max_epochs=args.max_epochs,
        callbacks=callbacks,
        enable_progress_bar=not args.tune,
    )
    device = trainer.strategy.root_device
    model = ShapesModel(device=device, **vars(args))
    trainer.fit(model, train_data, val_data)


def main(args):
    checkpoint_save = pl.callbacks.ModelCheckpoint(save_top_k=-1)
    train(args, [checkpoint_save])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--data_root", type=str, help="Path to event-based shape dataset")
    parser.add_argument("--timesteps", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--mode",
        default="single",
        choices=["default", "single", "single_triangle", "single_triangle_top"],
    )
    parser.add_argument("--network_delay", type=int, default=1)
    parser = ShapesModel.add_model_specific_args(parser)
    args = parser.parse_args()
    main(args)
