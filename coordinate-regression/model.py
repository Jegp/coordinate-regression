from typing import Dict, List, Tuple, Optional, Union
from functools import reduce
import torch
from torch.distributions.normal import Normal
import norse.torch as norse


def register_lif_parameters(
    p: norse.LIFParameters, scale: float, shape: torch.Size
) -> Tuple[norse.LIFParameters, torch.nn.ParameterList]:
    ts = Normal(p.tau_syn_inv.float(), scale).sample(shape)
    tm = Normal(p.tau_mem_inv.float(), scale).sample(shape)
    p = norse.LIFParameters(tau_syn_inv=ts, tau_mem_inv=tm, v_th=p.v_th)
    p_list = torch.nn.ParameterList([torch.nn.Parameter(x) for x in (ts, tm)])
    return p, p_list


def register_li_parameters(
    p: norse.LIParameters, scale: float, shape: torch.Size
) -> Tuple[norse.LIParameters, torch.nn.ParameterList]:
    ts = Normal(p.tau_syn_inv.float(), scale).sample(shape)
    tm = Normal(p.tau_mem_inv.float(), scale).sample(shape)
    p = norse.LIParameters(tau_syn_inv=ts, tau_mem_inv=tm)
    p_list = torch.nn.ParameterList([torch.nn.Parameter(x) for x in (ts, tm)])
    return p, p_list


class ANN(torch.nn.Module):
    def __init__(self, kernels: int, classes: int):
        super().__init__()

        # Set kernels
        if isinstance(kernels, int):
            k1 = kernels
            k2 = kernels * 2
            k3 = 3 * classes
        else:
            k1, k2, k3 = kernels

        self.module = torch.nn.Sequential(
            norse.Lift(torch.nn.Conv2d(1, k1, 7, bias=False, padding=3, stride=2)),
            norse.Lift(torch.nn.BatchNorm2d(k1)),
            norse.Lift(torch.nn.ReLU()),
            norse.Lift(torch.nn.Conv2d(k1, k2, 7, bias=False, padding=2, stride=2)),
            norse.Lift(torch.nn.BatchNorm2d(k2)),
            norse.Lift(torch.nn.ReLU()),
            norse.Lift(torch.nn.Conv2d(k2, k3, 5, bias=False, padding=1, stride=1)),
            norse.Lift(torch.nn.BatchNorm2d(k3)),
            norse.Lift(torch.nn.ReLU()),
            norse.Lift(torch.nn.ConvTranspose2d(k3, classes, 9, bias=False)),
            norse.Lift(torch.nn.ReLU()),
            torch.nn.Dropout(0.2),
        )
        self.out_shape = (69, 69)
        self.spikes = None

    def forward(self, x, s=None):
        return self.module(x), s


class ANNRF(torch.nn.Module):
    def __init__(self, rf_file: str, classes: int):
        super().__init__()

        rfs = torch.load(rf_file)
        ring_size = rfs[0].shape[-1]
        rf_modules = []
        for i in range(0, len(rfs), 2):
            r_all, r_rest = rfs[i], rfs[i + 1]
            rings_all = r_all.flatten(0, 1).unsqueeze(1)
            n_rings = len(rings_all)
            rings_rest = r_rest.flatten(0, 1).unsqueeze(1).repeat(1, n_rings, 1, 1)
            n2_rings = len(rings_rest)

            m = norse.Lift(
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        1, n_rings, ring_size, bias=False, padding=4, stride=2
                    ),
                    torch.nn.BatchNorm2d(n_rings),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.1),
                    torch.nn.Conv2d(
                        n_rings, n2_rings, ring_size, bias=False, padding=4, stride=1
                    ),
                    torch.nn.BatchNorm2d(n2_rings),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.1),
                )
            )
            # Load rings receptive fields
            m.lifted_module[0].weight = torch.nn.Parameter(
                rings_all, requires_grad=False
            )
            m.lifted_module[4].weight = torch.nn.Parameter(
                rings_rest,
                requires_grad=False,
            )
            rf_modules.append(m)
        self.rf_modules = torch.nn.ModuleList(rf_modules)

        self.module2 = norse.Lift(
            torch.nn.Sequential(
                torch.nn.Conv2d(
                    n2_rings, 3 * classes, 9, bias=False, padding=4, stride=1
                ),
                torch.nn.BatchNorm2d(3 * classes),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.ConvTranspose2d(3 * classes, classes, 9, bias=False),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
            )
        )

        self.out_shape = (136, 136)
        self.spikes = None

    def forward(self, x, s=None):
        with torch.no_grad():
            ys = []
            for m in self.rf_modules:
                y = m(x)
                ys.append(y)
        y = torch.concat(ys, dim=-3)
        return self.module2(y), None

    def forward_step(self, x, state=(None, None)):
        sr, sc = state
        if sc is None:
            sc = [None] * len(self.rf_modules)
        with torch.no_grad():
            ys = []
            ss = []
            for m, s in zip(self.rf_modules, sc):
                y, s = self.module(x, s)
                ys.append(y)
                ss.append(ss)
        y = torch.concat(ys, dim=-3)
        y, sc = self.classifier(y, sc)
        return y, (sr, sc)


class ShapesSNNLayer(torch.nn.Module):
    def __init__(
        self,
        li_p,
        lif_p,
        kernels: Union[int, torch.Size],
        classes: int,
        learn_parameters: bool = False,
    ):
        super().__init__()
        self.shapes = [(320, 240), (160, 120), (160, 120)]
        self.out_shape = (129, 129)

        # Set kernels
        if isinstance(kernels, int):
            k1 = kernels
            k2 = kernels * 2
            k3 = 3 * classes
        else:
            k1, k2, k3 = kernels

        # Neuron parameters
        if learn_parameters:
            sigma = 20
            self.lif_p1 = self.lif_p2 = self.lif_p3 = lif_p
            self.li_p, self.li_pl = register_li_parameters(
                li_p, sigma, [classes, *self.out_shape]
            )
        else:
            self.lif_p1 = self.lif_p2 = self.lif_p3 = lif_p
            self.li_p = li_p

        # fmt: off
        self.module = norse.SequentialState(
            *self.conv_block(1,  k1, 7, self.lif_p1, padding=3, stride=2),
            *self.conv_block(k1, k2, 7, self.lif_p2, padding=3, stride=2),
            *self.conv_block(k2, k3, 5, self.lif_p3, padding=2, stride=1),
            torch.nn.ConvTranspose2d(k3, classes, 9, bias=False),
            norse.LICell(p=self.li_p),
            torch.nn.Dropout(0.2),
        )
        # fmt: on

    @staticmethod
    def conv_block(c_in, c_out, kernel_size, p, **kwargs):
        block = [
            torch.nn.Conv2d(c_in, c_out, kernel_size, bias=False, **kwargs),
            torch.nn.BatchNorm2d(c_out),
            norse.LIFCell(p=p),
            torch.nn.Dropout(0.1),
        ]
        block[1].bias = None
        return block

    def forward(self, x, state=None):
        self.spikes = []  # Clear out spikes
        out = []
        for t in x:
            y, state = self.module(t, state)
            out.append(y)
        return torch.stack(out), state

    def forward_step(self, x, state=None):
        return self.module(x, state)


class ShapesSNNLayer(torch.nn.Module):
    def __init__(
        self,
        li_p,
        lif_p,
        kernels: Union[int, torch.Size],
        classes: int,
        learn_parameters: bool = False,
    ):
        super().__init__()
        self.shapes = [(128, 128), (128, 128), (128, 128)]
        self.out_shape = (129, 129)

        # Set kernels
        if isinstance(kernels, int):
            k1 = kernels
            k2 = kernels * 2
            k3 = 3 * classes
        else:
            k1, k2, k3 = kernels

        # Neuron parameters
        if learn_parameters:
            sigma = 20
            self.lif_p1 = self.lif_p2 = self.lif_p3 = lif_p
            self.li_p, self.li_pl = register_li_parameters(
                li_p, sigma, [classes, *self.out_shape]
            )
        else:
            self.lif_p1 = self.lif_p2 = self.lif_p3 = lif_p
            self.li_p = li_p

        # fmt: off
        self.module = norse.SequentialState(
            *self.conv_block(1,  k1, 9, self.lif_p1, padding=3, stride=2),
            *self.conv_block(k1, k2, 9, self.lif_p2, padding=3, stride=1),
            *self.conv_block(k2, k3, 9, self.lif_p3, padding=2, stride=1),
            torch.nn.ConvTranspose2d(k3, classes, 9, bias=False),
            norse.LICell(p=self.li_p),
            torch.nn.Dropout(0.2),
        )
        # fmt: on

    @staticmethod
    def conv_block(c_in, c_out, kernel_size, p, **kwargs):
        block = [
            torch.nn.Conv2d(c_in, c_out, kernel_size, bias=False, **kwargs),
            torch.nn.BatchNorm2d(c_out),
            norse.LIFCell(p=p),
            torch.nn.Dropout(0.1),
        ]
        block[1].bias = None
        return block

    def forward(self, x, state=None):
        self.spikes = []  # Clear out spikes
        out = []
        for t in x:
            y, state = self.module(t, state)
            out.append(y)
        return torch.stack(out), state

    def forward_step(self, x, state=None):
        return self.module(x, state)


class ShapesSNNRFLayer(torch.nn.Module):
    def __init__(self, li_p, lif_p, rf_file: str, classes: int):
        super().__init__()
        self.shapes = [(128, 128), (136, 136)]
        self.out_shape = self.shapes[-1]

        sigma = 10
        self.lif_p1 = self.lif_p2 = lif_p
        self.lif_p3, self.lif_p3l = register_lif_parameters(
            lif_p, sigma, [3 * classes, *self.shapes[-2]]
        )
        self.li_p, self.li_pl = register_li_parameters(
            li_p, sigma, [classes, *self.out_shape]
        )

        rfs = torch.load(rf_file)
        ring_size = rfs[0].shape[-1]
        rf_rest_size = rfs[-1].flatten(0, 1).shape[0]
        rf_modules = []
        for i in range(0, len(rfs), 2):
            r_all, r_rest = rfs[i], rfs[i + 1]
            rings_all = r_all.flatten(0, 1).unsqueeze(1)
            n_rings = len(rings_all)
            rings_rest = r_rest.flatten(0, 1).unsqueeze(1).repeat(1, n_rings, 1, 1)
            n2_rings = len(rings_rest)

            m = norse.SequentialState(
                *self.conv_block(
                    1, n_rings, ring_size, self.lif_p1, padding=4, stride=2
                ),
                *self.conv_block(
                    n_rings, n2_rings, ring_size, self.lif_p2, padding=4, stride=1
                ),
            )
            # Load rings receptive fields
            m[0].weight = torch.nn.Parameter(rings_all, requires_grad=False)
            m[4].weight = torch.nn.Parameter(rings_rest, requires_grad=False)
            rf_modules.append(m)
        self.rf_modules = torch.nn.ModuleList(rf_modules)

        self.classifier = norse.SequentialState(
            *self.conv_block(
                len(rf_modules) * rf_rest_size,
                3 * classes,
                9,
                self.lif_p3,
                padding=4,
                stride=1,
            ),
            torch.nn.ConvTranspose2d(3 * classes, classes, 9, bias=False),
            norse.LICell(p=self.li_p),
            torch.nn.Dropout(0.2),
        )

    @staticmethod
    def conv_block(c_in, c_out, kernel_size, p, **kwargs):
        block = [
            torch.nn.Conv2d(c_in, c_out, kernel_size, bias=False, **kwargs),
            torch.nn.BatchNorm2d(c_out),
            norse.LIFBoxCell(p=p),
            torch.nn.Dropout(0.1),
        ]
        block[1].bias = None
        return block

    def forward(self, x, state=(None, None)):
        self.spikes = []  # Clear out spikes
        out = []
        for t in x:
            y, state = self.forward_step(t, state)
            out.append(y)
        return torch.stack(out), state

    def forward_step(self, x, state=(None, None)):
        sr, sc = state
        if sr is None:
            sr = [None] * len(self.rf_modules)
        ys = []
        ss = []
        with torch.no_grad():
            for m, s in zip(self.rf_modules, sr):
                y, s = m(x, s)
                ys.append(y)
                ss.append(s)
        y = torch.concat(ys, dim=-3)
        y, sc = self.classifier(y, sc)
        return y, (ss, sc)
