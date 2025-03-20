# mypy: disable-error-code="has-type,attr-defined,assignment"
import torch
from caskade import Param


def e1e2_to_q(e1, e2):
    c = torch.clamp((e1**2 + e2**2).sqrt(), 0, 1)
    return (1 - c) / (1 + c)


def e1e2_to_phi(e1, e2):
    phi = 0.5 * torch.arctan2(e2, e1)
    return phi


def e1e2_to_qphi(e1, e2):
    q = e1e2_to_q(e1, e2)
    phi = e1e2_to_phi(e1, e2)
    return q, phi


def qphi_to_e1e2(q, phi):
    e1 = (1 - q) * torch.cos(2 * phi) / (1 + q)
    e2 = (1 - q) * torch.sin(2 * phi) / (1 + q)
    return e1, e2


def c1c2_to_q(c1, c2):
    c = (c1**2 + c2**2).sqrt()  # torch.clamp(, 0, 1)
    return 1 - c / (1 + c)


def c1c2_to_phi(c1, c2):
    phi = 0.5 * torch.arctan2(c2, c1)
    return phi


def c1c2_to_qphi(c1, c2):
    q = c1c2_to_q(c1, c2)
    phi = c1c2_to_phi(c1, c2)
    return q, phi


def qphi_to_c1c2(q, phi):
    c1 = (1 - q) * torch.cos(2 * phi) / q
    c2 = (1 - q) * torch.sin(2 * phi) / q
    return c1, c2


class Angle_Mixin:
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._angle_system = "q_phi"

    @property
    def angle_system(self) -> str:
        return self._angle_system

    @angle_system.setter
    def angle_system(self, value: str):
        if value not in ["q_phi", "e1_e2", "c1_c2"]:
            raise ValueError(
                "angle_system must be either 'q_phi', 'e1_e2', or 'c1_c2'."
            )

        if value == "e1_e2" and self._angle_system != "e1_e2":
            self.e1 = Param(
                "e1", shape=self.q.shape if self.q.static else (), units="unitless"
            )
            self.e2 = Param(
                "e2", shape=self.q.shape if self.q.static else (), units="unitless"
            )

            self.q = lambda p: e1e2_to_q(p["e1"].value, p["e2"].value)
            self.q.link(self.e1)
            self.q.link(self.e2)
            self.phi = lambda p: e1e2_to_phi(p["e1"].value, p["e2"].value)
            self.phi.link(self.e1)
            self.phi.link(self.e2)
            try:
                del self.c1
                del self.c2
            except AttributeError:
                pass
        elif value == "c1_c2" and self._angle_system != "c1_c2":
            self.c1 = Param(
                "c1", shape=self.q.shape if self.q.static else (), units="unitless"
            )
            self.c2 = Param(
                "c2", shape=self.q.shape if self.q.static else (), units="unitless"
            )

            self.q = lambda p: c1c2_to_q(p["c1"].value, p["c2"].value)
            self.q.link(self.c1)
            self.q.link(self.c2)
            self.phi = lambda p: c1c2_to_phi(p["c1"].value, p["c2"].value)
            self.phi.link(self.c1)
            self.phi.link(self.c2)
            try:
                del self.e1
                del self.e2
            except AttributeError:
                pass
        elif value == "q_phi" and self._angle_system != "q_phi":
            self.q = None
            self.phi = None
            try:
                del self.e1
                del self.e2
            except AttributeError:
                pass
            try:
                del self.c1
                del self.c2
            except AttributeError:
                pass

        self._angle_system = value
