from __future__ import annotations

__copyright__ = "Copyright (c) 2026 North-West University (NWU), South Africa."
__licence__ = "Apache 2.0; see LICENSE file for details."
__author__ = "Tian Theunissen, Coenraad Mouton"
__description__ = (
    "An AutoEncoder with configurable submodules."
)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torch import Tensor

import numpy as np
import torch
from torch import nn

from default_archs.LSTMv2 import Model as KnowItLSTM
from default_archs.CNN import Model as KnowItCNN
from helpers.logger import get_logger

logger = get_logger()

available_tasks = ("classification", )

HP_ranges_dict = {
    "latent_channels": range(2, 257, 2),
    "num_downsamples": range(0, 6),
    "depth_enc": range(1, 6),
    "depth_dec": range(1, 6),
    "width_enc": range(8, 257, 8),
    "width_dec": range(8, 257, 8),
    "kernel_size": (3, 5, 7, 9),
    "class_dim": range(4, 65, 4),
    "batchnorm": (True, False),
    "dropout": np.arange(0, 1.1, 0.1),
    "activations": ("ReLU", "LeakyReLU", "Tanh"),
}


# -----------------
# MAIN ARCHITECTURE
# -----------------

class Model(nn.Module):

    def __init__(
        self,
        input_dim: list[int],
        output_dim: list[int],
        task_name: str,
        *,
        manifold_type: str = None,
        encoder_type: str = 'LSTM',
        decoder_type: str = 'LSTM',
        is_class_conditional: bool = True,
        num_classes: int = None,
        class_dim: int = 16,
        component_compression: int = 2,
        timestep_compression: int = 2,
        encoder_hps: dict = {},
        decoder_hps: dict = {},
        manifold_hps: dict = {},
    ) -> None:
        super().__init__()

        # preliminaries
        self.t_size = input_dim[0]                                  # number of input and output timesteps
        self.c_size = input_dim[1]                                  # number of input and output components
        self.t_h_size = int(self.t_size / timestep_compression)     # number of timesteps after compression
        if self.t_h_size < 1:
            logger.error('Excessive timestep_compression resulting in t_h_size < 0.')
            exit(101)
        self.c_h_size = int(self.c_size / component_compression)    # number of components after compression (i.e. number of latent dimensions)
        if self.c_h_size < 1:
            logger.error('Excessive component_compression resulting in c_h_size < 0.')
            exit(101)
        if is_class_conditional and num_classes is None:
            logger.error('No num_classes specified for class_conditional.')
            exit(101)
        self.is_class_conditional = is_class_conditional
        self.num_classes = num_classes
        self.class_dim = class_dim
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.manifold_type = manifold_type

        if self.encoder_type == 'LSTM':
            enc_class = LSTMEncoder
        elif self.encoder_type == 'CNN':
            enc_class = CNNEncoder
        else:
            logger.error('Unknown encoder type %s' % self.encoder_type)
            exit(101)

        if self.decoder_type == 'LSTM':
            dec_class = LSTMDecoder
        elif self.decoder_type == 'CNN':
            dec_class = CNNDecoder
        else:
            logger.error('Unknown decoder type %s' % self.decoder_type)
            exit(101)

        if self.is_class_conditional:
            self.prepare_class_embeddings()
            self.encoder = enc_class(self.t_size, self.c_size + self.class_dim, self.t_h_size, self.c_h_size, encoder_hps)
            self.decoder = dec_class(self.t_size, self.c_size, self.t_h_size, self.c_h_size + self.class_dim, decoder_hps)
        else:
            self.encoder = enc_class(self.t_size, self.c_size, self.t_h_size, self.c_h_size, encoder_hps)
            self.decoder = dec_class(self.t_size, self.c_size, self.t_h_size, self.c_h_size, decoder_hps)


        if self.manifold_type is None:
            self.manifold = nn.Identity()
        elif self.manifold_type == 'Gaussian':
            self.manifold = GaussManifold(self.t_size, self.c_size, self.t_h_size, self.c_h_size, manifold_hps)
        elif self.manifold_type == 'Spherical':
            self.manifold = SphericalManifold(self.t_size, self.c_size, self.t_h_size, self.c_h_size, manifold_hps)
        else:
            logger.error('No manifold type matching %s.', self.manifold_type)
            exit(101)

    # MAIN FUNCTIONS

    def forward(self, x: Tensor, y: Tensor | None = None) -> Tensor:

        z = self.encode(x, y)

        v = self.manifold_proj(z)

        out = self.decode(v, y)

        return out

    def encode(self, x: Tensor, y: Tensor | None = None) -> Tensor:

        x = self.concat_class_embeddings(x, y)
        z = self.encoder(x)

        return z

    def manifold_proj(self, z: Tensor) -> Tensor:

        v = self.manifold(z)
        return v

    def decode(self, z: Tensor, y: Tensor | None = None) -> Tensor:

        z = self.concat_class_embeddings(z, y)
        out = self.decoder(z)
        return out

    # CLASS CONDITIONING FUNCTIONS

    def _get_class_embedding(self, y: Tensor | None, batch_size: int, device: torch.device) -> Tensor | None:

        if self.class_embedding is None:
            return None

        if y is None:
            return torch.zeros(batch_size, self.class_dim, device=device)

        if y.dim() > 1:
            if y.shape[-1] == self.num_classes:
                y = torch.argmax(y, dim=-1)
            else:
                logger.error('num_classes and given y tensor mismatch.')
                exit(101)

        y = y.long().view(batch_size)
        return self.class_embedding(y)

    def prepare_class_embeddings(self):

        if self.num_classes is None:
            logger.error('num_classes is None. Cannnot generate class embeddings.')
            exit(101)
        self.class_embedding = nn.Embedding(self.num_classes, self.class_dim)

    def sample_by_class(self, n_samples: int, class_id: int | Tensor) -> Tensor:

        if not self.is_class_conditional:
            logger.error('Encoder is not class conditional. Cannot sample by class.')
            exit(101)

        if self.num_classes <= 0:
            logger.error(
                "sample_by_class() called, but class conditioning is disabled (num_classes <= 0)."
            )
            exit(101)

        device = next(self.parameters()).device
        z = torch.normal(size=(n_samples, self.t_h_size, self.c_h_size), mean=0, std=1., device=device)


        z = self.manifold_proj(z)
        # z = self.manifold.spherifier(z)
        # z = z.view(-1, self.t_h_size, self.c_h_size)

        y = torch.zeros(size=(n_samples, self.num_classes), device=device)
        if isinstance(class_id, int):
            y[:, class_id] = 1.
        elif isinstance(class_id, list) and len(class_id) == n_samples:
            for c in range(len(class_id)):
                y[c, class_id[c]] = 1.
        else:
            logger.error('class_id must be int or list matching n_samples.')
            exit(101)

        return self.decode(z, y)

    def concat_class_embeddings(self, x, y):

        if self.is_class_conditional:
            batch_size = x.shape[0]
            class_emb = self._get_class_embedding(y, batch_size, x.device)
            if class_emb is not None:
                cond_map = class_emb.unsqueeze(1).expand(class_emb.shape[0], x.shape[1], class_emb.shape[1])
                x = torch.cat([x, cond_map], dim=-1)
        return x

    # LSTM FUNCTION WRAPPERS

    def force_reset(self) -> None:
        """ Wrapper for the underlying LSTM's corresponding function."""
        if hasattr(self.encoder, "force_reset"):
            self.encoder.force_reset()
        if hasattr(self.decoder, "force_reset"):
            self.decoder.force_reset()

    def get_internal_states(self) -> list:
        """ Wrapper for the underlying LSTM's corresponding function."""

        ret_list = []
        if hasattr(self.encoder, "get_internal_states"):
            ret_list.append(self.encoder.get_internal_states())
        if hasattr(self.decoder, "get_internal_states"):
            ret_list.append(self.decoder.get_internal_states())

        return ret_list

    def hard_set_states(self, ist_idx: Tensor) -> None:
        """ Wrapper for the underlying LSTM's corresponding function."""
        if hasattr(self.encoder, "hard_set_states"):
            self.encoder.hard_set_states(ist_idx)
        if hasattr(self.decoder, "hard_set_states"):
            self.decoder.hard_set_states(ist_idx)

    def update_states(self, ist_idx: Tensor, device: str) -> None:
        """ Wrapper for the underlying LSTM's corresponding function."""
        if hasattr(self.encoder, "update_states"):
            self.encoder.update_states(ist_idx, device)
        if hasattr(self.decoder, "update_states"):
            self.decoder.update_states(ist_idx, device)

    # GAUSSIAN MANIFOLD FUNCTION WRAPPERS

    def apply_kl_warmup(self, epoch: int) -> float:

        if hasattr(self.manifold, "apply_kl_warmup"):
            return self.manifold.apply_kl_warmup(epoch)
        else:
            return None

    def get_betakl(self) -> float:

        if hasattr(self.manifold, "get_betakl"):
            return self.manifold.get_betakl()
        else:
            return 0.0

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:

        if hasattr(self.manifold, "reparameterize"):
            return self.manifold.reparameterize(mu, logvar)
        else:
            return None

    def kl_loss(self) -> tuple[Tensor, Tensor]:

        if hasattr(self.manifold, "kl_loss"):
            return self.manifold.kl_loss()
        else:
            return None

    # SPHERICAL MANIFOLD FUNCTION WRAPPERS

    def get_Dv_NOISY(self, y) -> Tensor:

        v_NOISY = self.get_v_NOISY()
        if v_NOISY is not None:
            v_NOISY = self.concat_class_embeddings(v_NOISY, y)
            out = self.decoder(v_NOISY)
            return out
        else:
            return None

    def get_v_NOISY(self) -> Tensor:

        if hasattr(self.manifold, "get_v_NOISY"):
            return self.manifold.get_v_NOISY()
        else:
            return None

    def get_v(self) -> Tensor:

        if hasattr(self.manifold, "get_v"):
            return self.manifold.get_v()
        else:
            return None

    def get_lat_con_loss(self, y):

        v = self.get_v()
        Dv_NOISY = self.get_Dv_NOISY(y)

        if v is not None and Dv_NOISY is not None:
            EDv_NOISY = self.manifold.spherifier(self.encode(Dv_NOISY, y))
            lat_con_loss = 1. - torch.nn.CosineSimilarity()(v, EDv_NOISY).mean()
            return lat_con_loss
        else:
            return None

    def get_beta_pix_con(self) -> Tensor:

        if hasattr(self.manifold, "get_beta_pix_con"):
            return self.manifold.get_beta_pix_con()
        else:
            return None

    def get_beta_lat_con(self) -> Tensor:

        if hasattr(self.manifold, "get_beta_lat_con"):
            return self.manifold.get_beta_lat_con()
        else:
            return None

# -------
# ENCODER
# -------

# all encoders must map [b, t_size, c_size] -> [b, t_h_size, c_h_size]

class LSTMEncoder(nn.Module):
    def __init__(self, t_size: int, c_size: int, t_h_size: int, c_h_size: int, hps: dict) -> None:
        super().__init__()

        # preliminaries
        self.t_size = t_size
        self.c_size = c_size
        self.t_h_size = t_h_size
        self.c_h_size = c_h_size

        self.lstm = SubLSTM(in_dim=self.c_size, out_dim=self.c_h_size, hps=hps)

        if self.t_h_size > self.t_size:
            self.lstm_expansion_decoder = SubLSTM(in_dim=self.c_h_size, out_dim=self.c_h_size, hps=hps)

    def forward(self, x):

        z = self.lstm(x)

        if self.t_h_size < self.t_size:
            z = z[:, -self.t_h_size:, :]
        elif self.t_h_size > self.t_size:
            steps_to_expand = self.t_h_size - self.t_size
            states = self.lstm.get_internal_states()
            states = [item for sublist in states for item in sublist]
            exp_dec_input = z[:, -1:, :]
            self.lstm_expansion_decoder._overwrite_internal_states(states)
            extensions = []
            for _ in range(steps_to_expand):
                exp_dec_input = self.lstm_expansion_decoder(exp_dec_input)
                extensions.append(exp_dec_input)
            extra = torch.cat(extensions, dim=1)
            z = torch.cat([z, extra], dim=1)

        return z

    def force_reset(self) -> None:
        """ Wrapper for the underlying LSTM's corresponding function."""
        self.lstm.force_reset()
        if self.t_h_size > self.t_size:
            self.lstm_expansion_decoder.force_reset()

    def get_internal_states(self) -> list:
        """ Wrapper for the underlying LSTM's corresponding function."""
        return self.lstm.get_internal_states()

    def hard_set_states(self, ist_idx: Tensor) -> None:
        """ Wrapper for the underlying LSTM's corresponding function."""
        self.lstm.hard_set_states(ist_idx)

    def update_states(self, ist_idx: Tensor, device: str) -> None:
        """ Wrapper for the underlying LSTM's corresponding function."""
        self.lstm.update_states(ist_idx, device)

class CNNEncoder(nn.Module):
    def __init__(self, t_size: int, c_size: int, t_h_size: int, c_h_size: int, hps: dict) -> None:
        super().__init__()

        # preliminaries
        self.t_size = t_size
        self.c_size = c_size
        self.t_h_size = t_h_size
        self.c_h_size = c_h_size

        self.cnn = SubCNN(in_dim=self.c_size, out_dim=self.c_h_size, hps=hps)

        if self.t_h_size != self.t_size:
            self.cnn_expansion_decoder = ResampleModule(self.c_h_size, self.t_size, self.t_h_size)

    def forward(self, x):

        z = self.cnn(x)

        if self.t_h_size != self.t_size:
            z = self.cnn_expansion_decoder(z)

        return z

# --------
# MANIFOLD
# --------

# all manifold must map [b, t_h_size, c_h_size] -> [b, t_h_size, c_h_size]

class GaussManifold(nn.Module):
    def __init__(self, t_size: int, c_size: int, t_h_size: int, c_h_size: int,
                 hps: dict) -> None:
        super().__init__()

        # preliminaries
        self.t_size = t_size
        self.c_size = c_size
        self.t_h_size = t_h_size
        self.c_h_size = c_h_size

        self.kl_warmup_epochs = hps['kl_warmup_epochs'] if 'kl_warmup_epochs' in hps else 0
        self.beta_kl = hps['beta_kl'] if 'beta_kl' in hps else 0
        self.current_beta_kl = 0.0 if self.kl_warmup_epochs > 0 else self.beta_kl

        self.to_mu = nn.Conv1d(self.c_h_size, self.c_h_size, kernel_size=1)
        self.to_logvar = nn.Conv1d(self.c_h_size, self.c_h_size, kernel_size=1)

        self._last_mu: Tensor | None = None
        self._last_logvar: Tensor | None = None

    def forward(self, z: Tensor) -> Tensor:

        z = z.permute(0, 2, 1)

        mu = self.to_mu(z)
        logvar = self.to_logvar(z)

        v = self.reparameterize(mu, logvar)

        v = v.permute(0, 2, 1)

        self._last_mu = mu
        self._last_logvar = logvar

        return v

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick for latent tensors [B, C, T]."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_loss(self) -> tuple[Tensor, Tensor]:
        """
        Return weighted and unweighted KL divergence.

        Since mu/logvar are tensors [B, C, T], this KL is averaged over batch,
        channels, and latent time positions.
        """
        if self._last_mu is None or self._last_logvar is None:
            logger.error("kl_loss() called before forward().")
            exit(101)

        mu, logvar = self._last_mu, self._last_logvar
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        kl_weighted = self.current_beta_kl * kl
        return kl_weighted, kl

    def get_betakl(self) -> float:
        return self.current_beta_kl

    def apply_kl_warmup(self, epoch: int) -> float:
        """Linearly warm beta_kl from 0 to beta_kl over kl_warmup_epochs."""
        if self.kl_warmup_epochs <= 0:
            self.current_beta_kl = self.beta_kl
            return self.current_beta_kl

        frac = min(1.0, float(epoch + 1) / float(self.kl_warmup_epochs))
        self.current_beta_kl = min(self.beta_kl, self.beta_kl * frac)
        return self.current_beta_kl

class SphericalManifold(nn.Module):
    def __init__(self, t_size: int, c_size: int, t_h_size: int, c_h_size: int, hps: dict) -> None:
        super().__init__()

        # preliminaries
        self.t_size = t_size
        self.c_size = c_size
        self.t_h_size = t_h_size
        self.c_h_size = c_h_size
        self.sigma_max = hps['sigma_max'] if 'sigma_max' in hps else 10.0
        self.beta_lat_con = hps['beta_lat_con'] if 'beta_lat_con' in hps else 1.0
        self.beta_pix_con = hps['beta_pix_con'] if 'beta_pix_con' in hps else 1.0

        self.spherifier = SphericalNorm()

        self.v_NOISY = None
        self.v = None

    def forward(self, z: Tensor) -> Tensor:

        # spherify the incoming tensor
        v = self.spherifier(z)
        self.v = v

        device = v.device

        # add jittered noise
        e = self._random_noise_vector(v.shape, device)
        r = self._random_scalar(device)
        v_NOISY = v + r * self.sigma_max * e
        # project back onto sphere
        v_NOISY = self.spherifier(v_NOISY)
        # reshape into original view
        self.v_NOISY = v_NOISY.view(-1, self.t_h_size, self.c_h_size)

        # create small-noise alternative
        s = self._random_smaller_scalar(device)
        v_noisy = v + s * r * self.sigma_max * e
        v_noisy = self.spherifier(v_noisy)
        v_noisy = v_noisy.view(-1, self.t_h_size, self.c_h_size)

        return v_noisy

    def get_v_NOISY(self) -> Tensor:
        return self.v_NOISY

    def get_v(self) -> Tensor:
        return self.v

    def get_beta_lat_con(self) -> float:
        return self.beta_lat_con

    def get_beta_pix_con(self) -> float:
        return self.beta_pix_con

    @staticmethod
    def _random_noise_vector(shape, device):
        e = torch.normal(mean=0.0, std=1.0, size=shape, device=device)
        return e

    @staticmethod
    def _random_scalar(device):
        r = torch.rand(1, device=device)
        return r

    @staticmethod
    def _random_smaller_scalar(device):
        s = torch.rand(1, device=device) * 0.5
        return s

# -------
# DECODER
# -------

# all decoders must map [b, t_h_size, c_h_size] -> [b, t_size, c_size]

class LSTMDecoder(nn.Module):
    def __init__(self, t_size: int, c_size: int, t_h_size: int, c_h_size: int, hps: dict) -> None:
        super().__init__()

        # preliminaries
        self.t_size = t_size
        self.c_size = c_size
        self.t_h_size = t_h_size
        self.c_h_size = c_h_size

        self.lstm = SubLSTM(in_dim=self.c_h_size, out_dim=self.c_size, hps=hps)

        if self.t_h_size < self.t_size:
            self.lstm_expansion_decoder = SubLSTM(in_dim=self.c_size, out_dim=self.c_size, hps=hps)

    def forward(self, x):

        z = self.lstm(x)

        if self.t_h_size > self.t_size:
            z = z[:, -self.t_size:, :]
        elif self.t_h_size < self.t_size:
            steps_to_expand = self.t_size - self.t_h_size
            states = self.lstm.get_internal_states()
            states = [item for sublist in states for item in sublist]
            exp_dec_input = z[:, -1:, :]
            self.lstm_expansion_decoder._overwrite_internal_states(states)
            extensions = []
            for _ in range(steps_to_expand):
                exp_dec_input = self.lstm_expansion_decoder(exp_dec_input)
                extensions.append(exp_dec_input)
            extra = torch.cat(extensions, dim=1)
            z = torch.cat([z, extra], dim=1)

        return z

    def force_reset(self) -> None:
        """ Wrapper for the underlying LSTM's corresponding function."""
        self.lstm.force_reset()
        if self.t_h_size > self.t_size:
            self.lstm_expansion_decoder.force_reset()

    def get_internal_states(self) -> list:
        """ Wrapper for the underlying LSTM's corresponding function."""
        return self.lstm.get_internal_states()

    def hard_set_states(self, ist_idx: Tensor) -> None:
        """ Wrapper for the underlying LSTM's corresponding function."""
        self.lstm.hard_set_states(ist_idx)

    def update_states(self, ist_idx: Tensor, device: str) -> None:
        """ Wrapper for the underlying LSTM's corresponding function."""
        self.lstm.update_states(ist_idx, device)

class CNNDecoder(nn.Module):
    def __init__(self, t_size: int, c_size: int, t_h_size: int, c_h_size: int, hps: dict) -> None:
        super().__init__()

        # preliminaries
        self.t_size = t_size
        self.c_size = c_size
        self.t_h_size = t_h_size
        self.c_h_size = c_h_size

        self.cnn = SubCNN(in_dim=self.c_h_size, out_dim=self.c_size, hps=hps)

        if self.t_h_size != self.t_size:
            self.cnn_expansion_decoder = ResampleModule(self.c_size, self.t_h_size, self.t_size)

    def forward(self, x):

        z = self.cnn(x)

        if self.t_h_size != self.t_size:
            z = self.cnn_expansion_decoder(z)

        return z

# -----------
# SUB-MODULES
# -----------

class SubLSTM(KnowItLSTM):
    def __init__(self, in_dim: int, out_dim: int, hps: dict) -> None:
        super().__init__(input_dim=[1, in_dim],
                               output_dim=[1, out_dim],
                               task_name='vl_regression',
                               stateful=False,
                               depth= hps['depth'] if 'depth' in hps else 1,
                               dropout= hps ['dropout'] if 'dropout' in hps else 0.0,
                               width= hps['width'] if 'width' in hps else 256,
                               hc_init_method= hps['hc_init_method'] if 'hc_init_method' in hps else 'zeros',
                               layernorm= hps['layernorm'] if 'layernorm' in hps else True,
                               bidirectional= hps['bidirectional'] if 'bidirectional' in hps else False,
                               residual=False)

class SubCNN(KnowItCNN):
    def __init__(self, in_dim: int, out_dim: int, hps: dict) -> None:
        super().__init__(input_dim=[1, in_dim],
                         output_dim=[1, out_dim],
                         task_name='vl_regression',
                         depth=hps['depth'] if 'depth' in hps else 3,
                         num_filters=hps['num_filters'] if 'depth' in hps else 64,
                         kernel_size=hps['kernel_size'] if 'kernel_size' in hps else 3,
                         normalization=hps['normalization'] if 'normalization' in hps else 'batch',
                         dropout=hps['dropout'] if 'dropout' in hps else 0.0,
                         residual_connect=hps['residual_connect'] if 'residual_connect' in hps else True,
                         dilation_base=hps['dilation_base'] if 'dilation_base' in hps else 2,
                         )

class ResampleModule(nn.Module):

    # generated by Claude. To check!

    def __init__(self, c, b, d):
        super().__init__()

        if d >= b:
            # Upsample: b -> d
            self.conv = nn.ConvTranspose1d(
                in_channels=c,
                out_channels=c,
                kernel_size=d - b + 1,
                stride=1,
                padding=0
            )
        else:
            # Downsample: b -> d
            self.conv = nn.Conv1d(
                in_channels=c,
                out_channels=c,
                kernel_size=b - d + 1,
                stride=1,
                padding=0
            )

    def forward(self, x):
        # x: (a, b, c) -> permute to (a, c, b) for conv
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)  # back to (a, d, c)
        return x

class SphericalNorm(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, z):
        # Flatten [B, T, C] -> [B, T*C]
        x = z.flatten(1)

        # Standard RMS calculation
        # Each row (batch element) is normalized independently
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        # This results in a vector with L2 norm = sqrt(dim)
        return x / rms
