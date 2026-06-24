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

import functools
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from default_archs.LSTMv2 import Model as KnowItLSTM
from torch.nn.utils.parametrizations import weight_norm
from default_archs.CNN import Model as KnowItCNN, init_mod
from default_archs.MLP import Model as KnowItMLP
from helpers.logger import get_logger

logger = get_logger()

available_tasks = ("classification", "regression")

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


def _parse_rec_loss(rec_loss):
    """Return a callable loss function from a string or single-key dict.

    Looks up the function by name in ``torch.nn.functional``.  A dict value
    supplies keyword arguments that are frozen via ``functools.partial``.

    Examples
    --------
    ``'mse_loss'``                         → ``F.mse_loss``
    ``{'huber_loss': {'delta': 1.0}}``     → ``partial(F.huber_loss, delta=1.0)``
    """
    if isinstance(rec_loss, str):
        fn = getattr(F, rec_loss, None)
        if fn is None:
            logger.error("Unknown reconstruction loss '%s'. Must be a torch.nn.functional function.", rec_loss)
            exit(101)
        return fn
    if isinstance(rec_loss, dict):
        if len(rec_loss) != 1:
            logger.error("rec_loss dict must have exactly one key, got %d.", len(rec_loss))
            exit(101)
        name, kwargs = next(iter(rec_loss.items()))
        fn = getattr(F, name, None)
        if fn is None:
            logger.error("Unknown reconstruction loss '%s'. Must be a torch.nn.functional function.", name)
            exit(101)
        return functools.partial(fn, **kwargs)
    logger.error("rec_loss must be a string or dict, got %s.", type(rec_loss).__name__)
    exit(101)


# -----------------
# MAIN ARCHITECTURE
# -----------------

class Model(nn.Module):
    """Configurable auto-encoder for time series with optional class conditioning and manifold.

    Accepts KnowIt-style ``input_dim`` / ``output_dim`` / ``task_name`` arguments
    so that it can be registered as a KnowIt architecture.  Loss computation
    (reconstruction + any manifold-specific terms) is handled by
    :meth:`custom_loss_compute`, which is called by ``PLModel_custom`` during
    training instead of the standard KnowIt loss pipeline.

    The three-stage forward pass is: **encode** → **manifold projection** →
    **decode**.  The first and last stages support class conditioning via
    ``enc_conditioning_mode`` and ``dec_conditioning_mode`` (see parameter descriptions).

    Parameters
    ----------
    input_dim : list[int]
        ``[T, C]`` — number of timesteps and components in the input.
    output_dim : list[int]
        Accepted for KnowIt API compatibility; not used internally (the
        reconstruction target has the same shape as the input).
    task_name : str
        Accepted for KnowIt API compatibility; not used internally.
    manifold_type : str | None
        Latent manifold regularisation.  ``'Gaussian'`` applies VAE-style KL
        divergence via :class:`GaussManifold`; ``'Spherical'`` applies
        spherical consistency losses via :class:`SphericalManifold`; ``None``
        uses a plain AE with no regularisation.
    encoder_type : str
        Backbone for the encoder: ``'LSTM'`` (:class:`LSTMEncoder`),
        ``'CNN'`` (:class:`CNNEncoder`), ``'CNN2D'`` (:class:`CNN2DEncoder`),
        or ``'MLP'`` (:class:`MLPEncoder`).
    decoder_type : str
        Backbone for the decoder: ``'LSTM'`` (:class:`LSTMDecoder`),
        ``'CNN'`` (:class:`CNNDecoder`), ``'CNN2D'`` (:class:`CNN2DDecoder`),
        or ``'MLP'`` (:class:`MLPDecoder`).
    enc_conditioning_mode : str | dict | None
        How the encoder is conditioned on the class label.  ``'class_concatenate'``
        appends the class embedding to the input channels; ``'class_film'`` applies
        FiLM modulation (scale + shift) to the input; ``{'class_aux_loss': {'beta': 1.0}}``
        (dict form) adds a cross-entropy classification head on the encoder input without
        modifying the forward pass; ``None`` disables encoder conditioning.
    dec_conditioning_mode : str | dict | None
        Same options as ``enc_conditioning_mode``, applied to the latent before
        decoding.  ``'class_aux_loss'`` here classifies the manifold-projected
        latent, which is the more common use case.
    num_classes : int | None
        Number of distinct classes.  Required when either conditioning mode is not ``None``.
    class_dim : int
        Embedding dimension for each class label.
    component_compression : int
        Factor by which the component dimension is reduced in the latent
        (``c_h_size = c_size // component_compression``).
    timestep_compression : int
        Factor by which the time dimension is reduced in the latent
        (``t_h_size = t_size // timestep_compression``).
    encoder_hps : dict | None
        Hyper-parameters forwarded to the encoder sub-module.  See
        :class:`SubLSTM` / :class:`SubCNN` for accepted keys.
    decoder_hps : dict | None
        Hyper-parameters forwarded to the decoder sub-module.
    manifold_hps : dict | None
        Hyper-parameters forwarded to the manifold sub-module.  See
        :class:`GaussManifold` / :class:`SphericalManifold` for accepted keys.
    rec_loss : str | dict
        Reconstruction loss function.  Either a string naming a
        ``torch.nn.functional`` function (e.g. ``'mse_loss'``) or a single-key
        dict mapping the function name to its kwargs
        (e.g. ``{'huber_loss': {'delta': 1.0}}``).  Default ``'mse_loss'``.
    """

    def __init__(
        self,
        input_dim: list[int],
        output_dim: list[int],
        task_name: str,
        *,
        manifold_type: str | None = None,
        encoder_type: str = 'LSTM',
        decoder_type: str = 'LSTM',
        enc_conditioning_mode: str | dict | None = None,
        dec_conditioning_mode: str | dict | None = None,
        num_classes: int | None = None,
        class_dim: int = 16,
        component_compression: int = 2,
        timestep_compression: int = 2,
        encoder_hps: dict | None = None,
        decoder_hps: dict | None = None,
        manifold_hps: dict | None = None,
        rec_loss: str | dict = 'mse_loss',
    ) -> None:
        super().__init__()

        if encoder_hps is None:
            encoder_hps = {}
        if decoder_hps is None:
            decoder_hps = {}
        if manifold_hps is None:
            manifold_hps = {}

        # preliminaries
        self.t_size = input_dim[0]                                  # number of input and output timesteps
        self.c_size = input_dim[1]                                  # number of input and output components
        self.t_h_size = int(self.t_size / timestep_compression)     # number of timesteps in the latent
        if self.t_h_size < 1:
            logger.error('Excessive timestep_compression resulting in t_h_size < 1.')
            exit(101)
        self.c_h_size = int(self.c_size / component_compression)    # number of components in the latent
        if self.c_h_size < 1:
            logger.error('Excessive component_compression resulting in c_h_size < 1.')
            exit(101)
        if (enc_conditioning_mode is not None or dec_conditioning_mode is not None) and num_classes is None:
            logger.error('No num_classes specified for class conditioning.')
            exit(101)
        self.num_classes = num_classes
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.manifold_type = manifold_type
        self.rec_loss_f = _parse_rec_loss(rec_loss)

        if self.encoder_type == 'LSTM':
            enc_class = LSTMEncoder
        elif self.encoder_type == 'CNN':
            enc_class = CNNEncoder
        elif self.encoder_type == 'CNN2D':
            enc_class = CNN2DEncoder
        elif self.encoder_type == 'MLP':
            enc_class = MLPEncoder
        else:
            logger.error('Unknown encoder type %s' % self.encoder_type)
            exit(101)

        if self.decoder_type == 'LSTM':
            dec_class = LSTMDecoder
        elif self.decoder_type == 'CNN':
            dec_class = CNNDecoder
        elif self.decoder_type == 'CNN2D':
            dec_class = CNN2DDecoder
        elif self.decoder_type == 'MLP':
            dec_class = MLPDecoder
        else:
            logger.error('Unknown decoder type %s' % self.decoder_type)
            exit(101)

        self.enc_conditioner = Conditioner(self.t_size, self.c_size, num_classes, class_dim, enc_conditioning_mode)
        self.dec_conditioner = Conditioner(self.t_h_size, self.c_h_size, num_classes, class_dim, dec_conditioning_mode)
        _, enc_in = self.enc_conditioner.get_out_dim()
        _, dec_in = self.dec_conditioner.get_out_dim()
        self.encoder = enc_class(self.t_size, enc_in, self.t_h_size, self.c_h_size, encoder_hps)
        self.decoder = dec_class(self.t_size, self.c_size, self.t_h_size, dec_in, decoder_hps)

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
        """Encode, project through the manifold, and decode.

        Parameters
        ----------
        x : Tensor
            Input time series of shape ``[B, T, C]``.
        y : Tensor | None
            Class labels of shape ``[B]`` (integer indices) or
            ``[B, num_classes]`` (one-hot).  Only used when
            ``enc_conditioning_mode`` or ``dec_conditioning_mode`` is not ``None``.

        Returns
        -------
        Tensor
            Reconstruction of shape ``[B, T, C]``.
        """
        z = self.encode(x, y)
        v = self.manifold_proj(z)
        out = self.decode(v, y)
        return out

    def encode(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        """Optionally condition the input and run the encoder.

        Parameters
        ----------
        x : Tensor
            Input of shape ``[B, T, C]``.
        y : Tensor | None
            Class labels; forwarded to the encoder :class:`Conditioner`.

        Returns
        -------
        Tensor
            Latent representation of shape ``[B, t_h_size, c_h_size]``.
        """
        x = self.enc_conditioner(x, y)
        z = self.encoder(x)
        return z

    def manifold_proj(self, z: Tensor) -> Tensor:
        """Project the latent through the manifold module.

        Parameters
        ----------
        z : Tensor
            Latent tensor of shape ``[B, t_h_size, c_h_size]``.

        Returns
        -------
        Tensor
            Manifold-projected latent of the same shape.
        """
        v = self.manifold(z)
        return v

    def decode(self, z: Tensor, y: Tensor | None = None) -> Tensor:
        """Optionally condition the latent and run the decoder.

        Parameters
        ----------
        z : Tensor
            Latent tensor of shape ``[B, t_h_size, c_h_size]``.
        y : Tensor | None
            Class labels; forwarded to the decoder :class:`Conditioner`.

        Returns
        -------
        Tensor
            Reconstruction of shape ``[B, T, C]``.
        """
        z = self.dec_conditioner(z, y)
        out = self.decoder(z)
        return out

    def custom_loss_compute(self, x: Tensor, y: Tensor | None) -> tuple[Tensor, dict]:
        """Compute the reconstruction loss plus any manifold-specific auxiliary losses.

        Called by ``PLModel_custom`` instead of the standard KnowIt loss
        pipeline.  The total loss is the sum of the pixel-wise reconstruction
        loss and any weighted auxiliary terms from the active manifold.

        Parameters
        ----------
        x : Tensor
            Input time series of shape ``[B, T, C]``, used as both the
            encoder input and the reconstruction target.
        y : Tensor | None
            Class labels forwarded to :meth:`forward` for class conditioning.

        Returns
        -------
        tuple[Tensor, dict]
            ``(loss, loss_log_metrics)`` where ``loss`` is the scalar total
            loss and ``loss_log_metrics`` is a dict of named scalar tensors
            suitable for logging (e.g. ``'rec_loss'``, ``'weighted_kl'``).
        """
        y_pred = self.forward(x, y)
        rec_loss = self.rec_loss_f(x, y_pred)

        loss = rec_loss
        loss_log_metrics = {'rec_loss': rec_loss}

        # aux classification loss — must come before get_Dv_NOISY to avoid cache overwrite
        if self.enc_conditioner.conditioning_mode == 'class_aux_loss' and y is not None:
            enc_aux = self.enc_conditioner.compute_aux_loss(y)
            loss = loss + self.enc_conditioner.get_beta_cls() * enc_aux
            loss_log_metrics['enc_cls_aux_loss'] = enc_aux
        if self.dec_conditioner.conditioning_mode == 'class_aux_loss' and y is not None:
            dec_aux = self.dec_conditioner.compute_aux_loss(y)
            loss = loss + self.dec_conditioner.get_beta_cls() * dec_aux
            loss_log_metrics['dec_cls_aux_loss'] = dec_aux

        if self.manifold_type == 'Gaussian':

            # loss = rec_loss + beta * kl

            weighted_kl, kl = self.kl_loss()
            loss_log_metrics['weighted_kl'] = weighted_kl
            loss_log_metrics['kl'] = kl
            loss_log_metrics['beta_kl'] = self.get_betakl()
            loss = loss + weighted_kl
        elif self.manifold_type == 'Spherical':

            # loss = rec_loss + beta_los_con * lat_con_loss + beta_pix_con * pix_con_loss

            Dv_NOISY = self.get_Dv_NOISY(y)
            lat_con_loss = self.get_lat_con_loss(y, Dv_NOISY=Dv_NOISY)
            pix_con_loss = self.rec_loss_f(Dv_NOISY, y_pred.detach())
            loss = loss + self.get_beta_lat_con() * lat_con_loss + self.get_beta_pix_con() * pix_con_loss
            loss_log_metrics['lat_con_loss'] = lat_con_loss
            loss_log_metrics['pix_con_loss'] = pix_con_loss
            loss_log_metrics['beta_lat_con'] = self.get_beta_lat_con()
            loss_log_metrics['beta_pix_con'] = self.get_beta_pix_con()
        else:
            pass

        return loss, loss_log_metrics

    # GENERATING FUNCTIONS

    def sample_by_class(self, n_samples: int, class_id: int | list[int], refinement_iterations: int | None = None) -> Tensor:
        """Sample from the latent prior conditioned on a specific class.

        Samples ``z ~ N(0, I)`` in latent space, projects through the manifold,
        and decodes with the requested class conditioning.  Optionally refines
        the output by repeatedly passing it back through the encode→manifold→decode
        cycle, keeping the class label fixed throughout.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        class_id : int | list[int]
            A single integer (all samples from the same class) or a list of
            integers of length ``n_samples`` (per-sample class assignment).
        refinement_iterations : int | None
            Number of encode→manifold→decode refinement passes applied after
            the initial decoding.  ``None`` (default) skips refinement.

        Returns
        -------
        Tensor
            Generated time series of shape ``[n_samples, T, C]``.
        """
        if self.dec_conditioner.conditioning_mode not in ('class_concatenate', 'class_film'):
            logger.error('dec_conditioning_mode must be class_concatenate or class_film to sample by class.')
            exit(101)

        if self.num_classes <= 0:
            logger.error(
                "sample_by_class() called, but class conditioning is disabled (num_classes <= 0)."
            )
            exit(101)

        device = next(self.parameters()).device
        z = torch.normal(size=(n_samples, self.t_h_size, self.c_h_size), mean=0., std=1., device=device)

        y = torch.zeros(size=(n_samples, self.num_classes), device=device)
        if isinstance(class_id, int):
            y[:, class_id] = 1.
        elif isinstance(class_id, list) and len(class_id) == n_samples:
            for c in range(len(class_id)):
                y[c, class_id[c]] = 1.
        else:
            logger.error('class_id must be int or list matching n_samples.')
            exit(101)

        z = self.manifold_proj(z)
        x_hat = self.decode(z, y)

        if refinement_iterations is not None:
            for _ in range(refinement_iterations):
                x_hat = self.encode(x_hat, y)
                x_hat = self.manifold_proj(x_hat)
                x_hat = self.decode(x_hat, y)

        return x_hat

    def sample(self, n_samples: int, refinement_iterations: int | None = None) -> Tensor:
        """Sample from the latent prior without class conditioning.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        refinement_iterations : int | None
            The number of times to pass the generated samples through another pass of the AE.

        Returns
        -------
        Tensor
            Generated time series of shape ``[n_samples, T, C]``.
        """
        device = next(self.parameters()).device
        z = torch.normal(size=(n_samples, self.t_h_size, self.c_h_size), mean=0, std=1., device=device)

        if self.manifold_type == "Spherical":
            z = self.manifold_proj(z)

        x_hat = self.decode(z)

        if refinement_iterations is not None:
            for _ in range(refinement_iterations):
                x_hat = self.encode(x_hat)
                x_hat = self.manifold_proj(x_hat)
                x_hat = self.decode(x_hat)

        return x_hat

    # LSTM FUNCTION WRAPPERS

    def force_reset(self) -> None:
        """Wrapper for the underlying LSTM's corresponding function."""
        if hasattr(self.encoder, "force_reset"):
            self.encoder.force_reset()
        if hasattr(self.decoder, "force_reset"):
            self.decoder.force_reset()

    def get_internal_states(self) -> list:
        """Wrapper for the underlying LSTM's corresponding function."""
        ret_list = []
        if hasattr(self.encoder, "get_internal_states"):
            ret_list.append(self.encoder.get_internal_states())
        if hasattr(self.decoder, "get_internal_states"):
            ret_list.append(self.decoder.get_internal_states())
        return ret_list

    def hard_set_states(self, ist_idx: Tensor) -> None:
        """Wrapper for the underlying LSTM's corresponding function."""
        if hasattr(self.encoder, "hard_set_states"):
            self.encoder.hard_set_states(ist_idx)
        if hasattr(self.decoder, "hard_set_states"):
            self.decoder.hard_set_states(ist_idx)

    def update_states(self, ist_idx: Tensor, device: str) -> None:
        """Wrapper for the underlying LSTM's corresponding function."""
        if hasattr(self.encoder, "update_states"):
            self.encoder.update_states(ist_idx, device)
        if hasattr(self.decoder, "update_states"):
            self.decoder.update_states(ist_idx, device)

    # GAUSSIAN MANIFOLD FUNCTION WRAPPERS

    def apply_kl_warmup(self, epoch: int, training: bool = True) -> float | None:
        """Wrapper: linearly warm the KL weight over training epochs."""
        if hasattr(self.manifold, "apply_kl_warmup"):
            return self.manifold.apply_kl_warmup(epoch, training)
        else:
            return None

    def get_betakl(self) -> float:
        """Wrapper: return the current KL weight (0.0 if no Gaussian manifold)."""
        if hasattr(self.manifold, "get_betakl"):
            return self.manifold.get_betakl()
        else:
            return 0.0

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor | None:
        """Wrapper: apply the reparameterization trick."""
        if hasattr(self.manifold, "reparameterize"):
            return self.manifold.reparameterize(mu, logvar)
        else:
            return None

    def kl_loss(self) -> tuple[Tensor, Tensor] | None:
        """Wrapper: return ``(weighted_kl, kl)`` from the Gaussian manifold."""
        if hasattr(self.manifold, "kl_loss"):
            return self.manifold.kl_loss()
        else:
            return None

    # SPHERICAL MANIFOLD FUNCTION WRAPPERS

    def apply_sphere_warmup(self, epoch: int, training: bool = True) -> tuple[float, float] | None:
        """Wrapper: linearly warm the spherical consistency loss weights."""
        if hasattr(self.manifold, "apply_sphere_warmup"):
            return self.manifold.apply_sphere_warmup(epoch, training)
        else:
            return None

    def get_Dv_NOISY(self, y: Tensor | None) -> Tensor | None:
        """Decode the noisy sphere projection stored by :class:`SphericalManifold`.

        Retrieves ``v_NOISY`` from the last forward pass, appends class
        embeddings, and passes it through the decoder.

        Parameters
        ----------
        y : Tensor | None
            Class labels for conditional decoding.

        Returns
        -------
        Tensor | None
            Decoded noisy latent of shape ``[B, T, C]``, or ``None`` if no
            Spherical manifold is active.
        """
        v_NOISY = self.get_v_NOISY()
        if v_NOISY is not None:
            v_NOISY = self.dec_conditioner(v_NOISY, y)
            out = self.decoder(v_NOISY)
            return out
        else:
            return None

    def get_v_NOISY(self) -> Tensor | None:
        """Wrapper: return the large-noise sphere projection from the last forward pass."""
        if hasattr(self.manifold, "get_v_NOISY"):
            return self.manifold.get_v_NOISY()
        else:
            return None

    def get_v(self) -> Tensor | None:
        """Wrapper: return the clean sphere projection from the last forward pass."""
        if hasattr(self.manifold, "get_v"):
            return self.manifold.get_v()
        else:
            return None

    def get_lat_con_loss(self, y: Tensor | None, Dv_NOISY: Tensor | None = None) -> Tensor | None:
        """Compute the latent consistency loss for the Spherical manifold.

        Encodes the decoded noisy latent ``D(v_NOISY)`` and measures its
        cosine distance from the clean sphere projection ``v``.  This
        penalises large movements on the sphere that cause inconsistent
        reconstructions.

        Parameters
        ----------
        y : Tensor | None
            Class labels used for both encoding and decoding.
        Dv_NOISY : Tensor | None
            Pre-computed ``D(v_NOISY)`` to avoid a redundant decoder pass.
            If ``None``, it is computed internally.

        Returns
        -------
        Tensor | None
            Scalar consistency loss in ``[0, 2]``, or ``None`` if the
            Spherical manifold is not active.
        """
        v = self.get_v()
        if Dv_NOISY is None:
            Dv_NOISY = self.get_Dv_NOISY(y)

        if v is not None and Dv_NOISY is not None:
            EDv_NOISY = self.manifold.spherifier(self.encode(Dv_NOISY, y))
            lat_con_loss = 1. - torch.nn.CosineSimilarity()(v, EDv_NOISY).mean()
            return lat_con_loss
        else:
            return None

    def get_beta_pix_con(self) -> float | None:
        """Wrapper: return the pixel-consistency loss weight."""
        if hasattr(self.manifold, "get_beta_pix_con"):
            return self.manifold.get_beta_pix_con()
        else:
            return None

    def get_beta_lat_con(self) -> float | None:
        """Wrapper: return the latent-consistency loss weight."""
        if hasattr(self.manifold, "get_beta_lat_con"):
            return self.manifold.get_beta_lat_con()
        else:
            return None

# -------
# ENCODER
# -------

# all encoders must map [b, t_size, c_size] -> [b, t_h_size, c_h_size]

class LSTMEncoder(nn.Module):
    """LSTM-based encoder that maps ``[B, t_size, c_size]`` → ``[B, t_h_size, c_h_size]``.

    When ``t_h_size < t_size`` the encoder keeps the last ``t_h_size`` outputs
    of the LSTM.  When ``t_h_size > t_size`` a second LSTM auto-regressively
    extends the sequence beyond the input length.

    Parameters
    ----------
    t_size : int
        Number of input timesteps.
    c_size : int
        Number of input channels (may include appended class embedding channels).
    t_h_size : int
        Number of latent timesteps.
    c_h_size : int
        Number of latent channels.
    hps : dict
        Hyper-parameters forwarded to :class:`SubLSTM`.
    """

    def __init__(self, t_size: int, c_size: int, t_h_size: int, c_h_size: int, hps: dict) -> None:
        super().__init__()

        self.t_size = t_size
        self.c_size = c_size
        self.t_h_size = t_h_size
        self.c_h_size = c_h_size

        self.lstm = SubLSTM(in_dim=self.c_size, out_dim=self.c_h_size, hps=hps)

        if self.t_h_size > self.t_size:
            self.lstm_expansion_decoder = SubLSTM(in_dim=self.c_h_size, out_dim=self.c_h_size, hps=hps)

    def forward(self, x: Tensor) -> Tensor:
        """Encode ``x`` to the latent time resolution.

        Parameters
        ----------
        x : Tensor
            Input of shape ``[B, t_size, c_size]``.

        Returns
        -------
        Tensor
            Latent of shape ``[B, t_h_size, c_h_size]``.
        """
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
        """Wrapper for the underlying LSTM's corresponding function."""
        self.lstm.force_reset()
        if self.t_h_size > self.t_size:
            self.lstm_expansion_decoder.force_reset()

    def get_internal_states(self) -> list:
        """Wrapper for the underlying LSTM's corresponding function."""
        return self.lstm.get_internal_states()

    def hard_set_states(self, ist_idx: Tensor) -> None:
        """Wrapper for the underlying LSTM's corresponding function."""
        self.lstm.hard_set_states(ist_idx)

    def update_states(self, ist_idx: Tensor, device: str) -> None:
        """Wrapper for the underlying LSTM's corresponding function."""
        self.lstm.update_states(ist_idx, device)


class CNNEncoder(nn.Module):
    """CNN-based encoder that maps ``[B, t_size, c_size]`` → ``[B, t_h_size, c_h_size]``.

    A :class:`ResampleModule` handles any mismatch between ``t_size`` and
    ``t_h_size``.

    Parameters
    ----------
    t_size : int
        Number of input timesteps.
    c_size : int
        Number of input channels.
    t_h_size : int
        Number of latent timesteps.
    c_h_size : int
        Number of latent channels.
    hps : dict
        Hyper-parameters forwarded to :class:`SubCNN`.
    """

    def __init__(self, t_size: int, c_size: int, t_h_size: int, c_h_size: int, hps: dict) -> None:
        super().__init__()

        self.t_size = t_size
        self.c_size = c_size
        self.t_h_size = t_h_size
        self.c_h_size = c_h_size

        self.cnn = SubCNN(in_dim=self.c_size, out_dim=self.c_h_size, hps=hps)

        if self.t_h_size != self.t_size:
            self.cnn_expansion_decoder = ResampleModule(self.c_h_size, self.t_size, self.t_h_size)

    def forward(self, x: Tensor) -> Tensor:
        """Encode ``x`` to the latent time resolution.

        Parameters
        ----------
        x : Tensor
            Input of shape ``[B, t_size, c_size]``.

        Returns
        -------
        Tensor
            Latent of shape ``[B, t_h_size, c_h_size]``.
        """
        z = self.cnn(x)

        if self.t_h_size != self.t_size:
            z = self.cnn_expansion_decoder(z)

        return z


class MLPEncoder(nn.Module):
    """MLP-based encoder that maps ``[B, t_size, c_size]`` → ``[B, t_h_size, c_h_size]``.

    Flattens the input to a vector of size ``t_size * c_size``, passes it
    through a :class:`SubMLP`, and reshapes the output to
    ``[B, t_h_size, c_h_size]``.  The full dimension change is handled
    internally by the MLP, so no separate :class:`ResampleModule` is needed.

    Parameters
    ----------
    t_size : int
        Number of input timesteps.
    c_size : int
        Number of input channels.
    t_h_size : int
        Number of latent timesteps.
    c_h_size : int
        Number of latent channels.
    hps : dict
        Hyper-parameters forwarded to :class:`SubMLP`.
    """

    def __init__(self, t_size: int, c_size: int, t_h_size: int, c_h_size: int, hps: dict) -> None:
        super().__init__()
        self.mlp = SubMLP(in_dim=[t_size, c_size], out_dim=[t_h_size, c_h_size], hps=hps)

    def forward(self, x: Tensor) -> Tensor:
        """Encode ``x`` to the latent shape.

        Parameters
        ----------
        x : Tensor
            Input of shape ``[B, t_size, c_size]``.

        Returns
        -------
        Tensor
            Latent of shape ``[B, t_h_size, c_h_size]``.
        """
        return self.mlp(x)


# --------
# MANIFOLD
# --------

# all manifolds must map [b, t_h_size, c_h_size] -> [b, t_h_size, c_h_size]

class GaussManifold(nn.Module):
    """VAE-style Gaussian manifold regulariser.

    Applies the reparameterization trick to the encoder output and tracks the
    KL divergence term.  Supports optional linear KL warm-up over training
    epochs to avoid posterior collapse early in training.

    Parameters
    ----------
    t_size : int
        Number of input timesteps (stored for reference).
    c_size : int
        Number of input channels (stored for reference).
    t_h_size : int
        Number of latent timesteps.
    c_h_size : int
        Number of latent channels.
    hps : dict
        Accepted keys:

        - ``'kl_warmup_epochs'`` (int, default 0): epoch at which ``beta_kl``
          reaches its target value (end of the linear ramp).
        - ``'kl_warmup_epochs_start'`` (int, default 0): epoch at which the
          linear ramp begins.  ``beta_kl`` is held at 0 before this epoch.
        - ``'beta_kl'`` (float, default 0): target KL weight.
    """

    def __init__(self, t_size: int, c_size: int, t_h_size: int, c_h_size: int,
                 hps: dict) -> None:
        super().__init__()

        self.t_size = t_size
        self.c_size = c_size
        self.t_h_size = t_h_size
        self.c_h_size = c_h_size

        self.kl_warmup_epochs = hps['kl_warmup_epochs'] if 'kl_warmup_epochs' in hps else 0
        self.kl_warmup_epochs_start = hps['kl_warmup_epochs_start'] if 'kl_warmup_epochs_start' in hps else 0
        self.beta_kl = hps['beta_kl'] if 'beta_kl' in hps else 0
        self.current_beta_kl = 0.0 if self.kl_warmup_epochs > 0 else self.beta_kl

        self.to_mu = nn.Conv1d(self.c_h_size, self.c_h_size, kernel_size=1)
        self.to_logvar = nn.Conv1d(self.c_h_size, self.c_h_size, kernel_size=1)

        self._last_mu: Tensor | None = None
        self._last_logvar: Tensor | None = None

    def forward(self, z: Tensor) -> Tensor:
        """Map encoder output to a reparameterized latent sample.

        Stores ``mu`` and ``logvar`` internally so :meth:`kl_loss` can be
        called afterwards without re-running the network.

        Parameters
        ----------
        z : Tensor
            Encoder output of shape ``[B, t_h_size, c_h_size]``.

        Returns
        -------
        Tensor
            Reparameterized sample of shape ``[B, t_h_size, c_h_size]``.
        """
        z = z.permute(0, 2, 1)

        mu = self.to_mu(z)
        logvar = self.to_logvar(z).clamp(-10., 10.)

        v = self.reparameterize(mu, logvar)

        v = v.permute(0, 2, 1)

        self._last_mu = mu
        self._last_logvar = logvar

        return v

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick for latent tensors ``[B, C, T]``."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_loss(self) -> tuple[Tensor, Tensor]:
        """Return weighted and unweighted KL divergence.

        Since ``mu`` and ``logvar`` are tensors of shape ``[B, C, T]``, the
        KL is averaged over batch, channels, and latent time positions.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(kl_weighted, kl)`` — both scalar tensors.
        """
        if self._last_mu is None or self._last_logvar is None:
            logger.error("kl_loss() called before forward().")
            exit(101)

        mu, logvar = self._last_mu, self._last_logvar
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        kl_weighted = self.current_beta_kl * kl
        return kl_weighted, kl

    def get_betakl(self) -> float:
        """Return the current KL weight."""
        return self.current_beta_kl

    def apply_kl_warmup(self, epoch: int, training: bool = True) -> float:
        """Linearly warm ``beta_kl`` from 0 to its target between ``kl_warmup_epochs_start`` and ``kl_warmup_epochs``.

        ``beta_kl`` is held at 0 before ``kl_warmup_epochs_start`` and at its
        target value after ``kl_warmup_epochs``.  During validation / test
        (``training=False``) the full ``beta_kl`` is applied so that logged
        metrics are comparable across runs.

        Parameters
        ----------
        epoch : int
            Current epoch index (0-based).
        training : bool
            If ``False``, sets ``current_beta_kl = beta_kl`` without ramping.

        Returns
        -------
        float
            The updated ``current_beta_kl``.
        """
        if training:
            if self.kl_warmup_epochs <= 0:
                self.current_beta_kl = self.beta_kl
                return self.current_beta_kl
            ramp_duration = self.kl_warmup_epochs - self.kl_warmup_epochs_start
            if ramp_duration <= 0 or epoch + 1 <= self.kl_warmup_epochs_start:
                frac = 0.0
            else:
                frac = min(1.0, float(epoch + 1 - self.kl_warmup_epochs_start) / float(ramp_duration))
            self.current_beta_kl = self.beta_kl * frac
        else:
            self.current_beta_kl = self.beta_kl

        return self.current_beta_kl


class SphericalManifold(nn.Module):
    """Spherical manifold regulariser using a contrastive sphere consistency approach.

    Projects the encoder output onto the unit hypersphere via
    :class:`SphericalNorm`, then generates two perturbed versions of the
    projection — a large-noise variant (``v_NOISY``) used for the
    pixel-consistency loss and a small-noise variant returned to the decoder.

    Parameters
    ----------
    t_size : int
        Number of input timesteps (stored for reference).
    c_size : int
        Number of input channels (stored for reference).
    t_h_size : int
        Number of latent timesteps.
    c_h_size : int
        Number of latent channels.
    hps : dict
        Accepted keys:

        - ``'sigma_max'`` (float, default 10.0): maximum noise scale for the base range.
        - ``'sigma_mix'`` (float | None, default None): upper noise scale for the high-angle
          mixing range.  Set to ``tan(85°) ≈ 11.43`` for small images.  ``None`` disables mixing.
        - ``'mix_prob'`` (float, default 0.0): probability of drawing σ from the high-angle
          range ``[sigma_max, sigma_mix]`` instead of the base range ``[0, sigma_max]``.
        - ``'beta_lat_con'`` (float, default 1.0): target latent-consistency loss weight.
        - ``'beta_pix_con'`` (float, default 1.0): target pixel-consistency loss weight.
        - ``'warmup_epochs'`` (int, default 0): number of epochs over which both betas are
          linearly ramped from 0 to their target values.  0 disables warmup.
    """

    def __init__(self, t_size: int, c_size: int, t_h_size: int, c_h_size: int, hps: dict) -> None:
        super().__init__()

        self.t_size = t_size
        self.c_size = c_size
        self.t_h_size = t_h_size
        self.c_h_size = c_h_size
        self.sigma_max = hps['sigma_max'] if 'sigma_max' in hps else 10.0
        self.sigma_mix = hps['sigma_mix'] if 'sigma_mix' in hps else None
        self.mix_prob = hps['mix_prob'] if 'mix_prob' in hps else 0.0
        self.beta_lat_con = hps['beta_lat_con'] if 'beta_lat_con' in hps else 1.0
        self.beta_pix_con = hps['beta_pix_con'] if 'beta_pix_con' in hps else 1.0
        self.warmup_epochs = hps['warmup_epochs'] if 'warmup_epochs' in hps else 0
        self.current_beta_lat_con = 0.0 if self.warmup_epochs > 0 else self.beta_lat_con
        self.current_beta_pix_con = 0.0 if self.warmup_epochs > 0 else self.beta_pix_con

        self.spherifier = SphericalNorm()

        self.v_NOISY: Tensor | None = None
        self.v: Tensor | None = None

    def forward(self, z: Tensor, noise_at_inference: bool = False) -> Tensor:
        """Project ``z`` onto the sphere and generate perturbed latents.

        Stores the clean sphere projection ``self.v`` (shape ``[B, T*C]``)
        and the large-noise projection ``self.v_NOISY`` (shape
        ``[B, t_h_size, c_h_size]``) for use in the consistency losses.

        Parameters
        ----------
        z : Tensor
            Encoder output of shape ``[B, t_h_size, c_h_size]``.
        noise_at_inference : bool
            Whether to add the training noise during inference in any case.
            Used for iterative refinement during generation. Default: ``False``.

        Returns
        -------
        Tensor
            Clean sphere projection of shape ``[B, t_h_size, c_h_size]`` when
            ``not self.training and not noise_at_inference``; otherwise the
            small-noise perturbed projection of the same shape.
        """
        
        # spherify the incoming tensor
        v = self.spherifier(z)
        self.v = v

        device = v.device

        # add jittered noise
        batch_size = v.shape[0]
        e = self._random_noise_vector(v.shape, device)
        sigma = self._sample_sigma(batch_size, device)
        v_NOISY = v + sigma * e
        # project back onto sphere
        v_NOISY = self.spherifier(v_NOISY)
        # reshape into original view
        self.v_NOISY = v_NOISY.view(-1, self.t_h_size, self.c_h_size)

        if not self.training and not noise_at_inference:
            return v.view(-1, self.t_h_size, self.c_h_size)

        # create small-noise alternative
        s = self._random_smaller_scalar(batch_size, device)
        v_noisy = v + s * sigma * e
        v_noisy = self.spherifier(v_noisy)
        v_noisy = v_noisy.view(-1, self.t_h_size, self.c_h_size)

        return v_noisy

    def get_v_NOISY(self) -> Tensor | None:
        """Return the large-noise sphere projection from the last forward pass."""
        return self.v_NOISY

    def get_v(self) -> Tensor | None:
        """Return the clean sphere projection from the last forward pass."""
        return self.v

    def get_beta_lat_con(self) -> float:
        """Return the current (possibly warmed-up) latent-consistency loss weight."""
        return self.current_beta_lat_con

    def get_beta_pix_con(self) -> float:
        """Return the current (possibly warmed-up) pixel-consistency loss weight."""
        return self.current_beta_pix_con

    def apply_sphere_warmup(self, epoch: int, training: bool = True) -> tuple[float, float]:
        """Linearly warm both betas from 0 to their targets over ``warmup_epochs``.

        During validation / test (``training=False``) the full target betas are
        applied so logged metrics are comparable across runs.

        Parameters
        ----------
        epoch : int
            Current epoch index (0-based).
        training : bool
            If ``False``, sets both current betas to their target values.

        Returns
        -------
        tuple[float, float]
            ``(current_beta_lat_con, current_beta_pix_con)``
        """
        if training:
            if self.warmup_epochs <= 0:
                self.current_beta_lat_con = self.beta_lat_con
                self.current_beta_pix_con = self.beta_pix_con
            else:
                frac = min(1.0, float(epoch + 1) / float(self.warmup_epochs))
                self.current_beta_lat_con = self.beta_lat_con * frac
                self.current_beta_pix_con = self.beta_pix_con * frac
        else:
            self.current_beta_lat_con = self.beta_lat_con
            self.current_beta_pix_con = self.beta_pix_con
        return self.current_beta_lat_con, self.current_beta_pix_con

    @staticmethod
    def _random_noise_vector(shape: tuple, device: torch.device) -> Tensor:
        """Sample a standard Gaussian noise vector."""
        return torch.normal(mean=0.0, std=1.0, size=shape, device=device)

    def _sample_sigma(self, batch_size: int, device: torch.device) -> Tensor:
        """Sample per-sample noise scale σ of shape ``[B, 1]``.

        With probability ``mix_prob``, draws σ from the high-angle range
        ``[sigma_max, sigma_mix]``; otherwise draws from ``[0, sigma_max]``.
        When ``sigma_mix`` is ``None`` or ``mix_prob`` is 0, always draws
        from the base range.
        """
        sigma = torch.rand(batch_size, 1, device=device) * self.sigma_max
        if self.sigma_mix is not None and self.mix_prob > 0.0:
            use_mix = torch.rand(batch_size, 1, device=device) < self.mix_prob
            sigma_high = self.sigma_max + torch.rand(batch_size, 1, device=device) * (self.sigma_mix - self.sigma_max)
            sigma = torch.where(use_mix, sigma_high, sigma)
        return sigma

    @staticmethod
    def _random_smaller_scalar(batch_size: int, device: torch.device) -> Tensor:
        """Sample a per-sample uniform random scalar in ``[0, 0.5)``, shape ``[B, 1]``."""
        return torch.rand(batch_size, 1, device=device) * 0.5


# -------
# DECODER
# -------

# all decoders must map [b, t_h_size, c_h_size] -> [b, t_size, c_size]

class LSTMDecoder(nn.Module):
    """LSTM-based decoder that maps ``[B, t_h_size, c_h_size]`` → ``[B, t_size, c_size]``.

    When ``t_h_size > t_size`` the decoder keeps the last ``t_size`` outputs of
    the LSTM.  When ``t_h_size < t_size`` a second LSTM auto-regressively
    extends the sequence to the required output length.

    Parameters
    ----------
    t_size : int
        Number of output timesteps.
    c_size : int
        Number of output channels.
    t_h_size : int
        Number of latent timesteps.
    c_h_size : int
        Number of latent channels (may include appended class embedding channels).
    hps : dict
        Hyper-parameters forwarded to :class:`SubLSTM`.
    """

    def __init__(self, t_size: int, c_size: int, t_h_size: int, c_h_size: int, hps: dict) -> None:
        super().__init__()

        self.t_size = t_size
        self.c_size = c_size
        self.t_h_size = t_h_size
        self.c_h_size = c_h_size

        self.lstm = SubLSTM(in_dim=self.c_h_size, out_dim=self.c_size, hps=hps)

        if self.t_h_size < self.t_size:
            self.lstm_expansion_decoder = SubLSTM(in_dim=self.c_size, out_dim=self.c_size, hps=hps)

    def forward(self, x: Tensor) -> Tensor:
        """Decode the latent ``x`` to the output time resolution.

        Parameters
        ----------
        x : Tensor
            Latent of shape ``[B, t_h_size, c_h_size]``.

        Returns
        -------
        Tensor
            Reconstruction of shape ``[B, t_size, c_size]``.
        """
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
        """Wrapper for the underlying LSTM's corresponding function."""
        self.lstm.force_reset()
        if self.t_h_size < self.t_size:
            self.lstm_expansion_decoder.force_reset()

    def get_internal_states(self) -> list:
        """Wrapper for the underlying LSTM's corresponding function."""
        return self.lstm.get_internal_states()

    def hard_set_states(self, ist_idx: Tensor) -> None:
        """Wrapper for the underlying LSTM's corresponding function."""
        self.lstm.hard_set_states(ist_idx)

    def update_states(self, ist_idx: Tensor, device: str) -> None:
        """Wrapper for the underlying LSTM's corresponding function."""
        self.lstm.update_states(ist_idx, device)


class CNNDecoder(nn.Module):
    """CNN-based decoder that maps ``[B, t_h_size, c_h_size]`` → ``[B, t_size, c_size]``.

    A :class:`ResampleModule` handles any mismatch between ``t_h_size`` and
    ``t_size``.

    Parameters
    ----------
    t_size : int
        Number of output timesteps.
    c_size : int
        Number of output channels.
    t_h_size : int
        Number of latent timesteps.
    c_h_size : int
        Number of latent channels.
    hps : dict
        Hyper-parameters forwarded to :class:`SubCNN`.
    """

    def __init__(self, t_size: int, c_size: int, t_h_size: int, c_h_size: int, hps: dict) -> None:
        super().__init__()

        self.t_size = t_size
        self.c_size = c_size
        self.t_h_size = t_h_size
        self.c_h_size = c_h_size

        self.cnn = SubCNN(in_dim=self.c_h_size, out_dim=self.c_size, hps=hps)

        if self.t_h_size != self.t_size:
            self.cnn_expansion_decoder = ResampleModule(self.c_size, self.t_h_size, self.t_size)

    def forward(self, x: Tensor) -> Tensor:
        """Decode the latent ``x`` to the output time resolution.

        Parameters
        ----------
        x : Tensor
            Latent of shape ``[B, t_h_size, c_h_size]``.

        Returns
        -------
        Tensor
            Reconstruction of shape ``[B, t_size, c_size]``.
        """
        z = self.cnn(x)

        if self.t_h_size != self.t_size:
            z = self.cnn_expansion_decoder(z)

        return z


class MLPDecoder(nn.Module):
    """MLP-based decoder that maps ``[B, t_h_size, c_h_size]`` → ``[B, t_size, c_size]``.

    Flattens the latent to a vector of size ``t_h_size * c_h_size``, passes it
    through a :class:`SubMLP`, and reshapes the output to
    ``[B, t_size, c_size]``.  The full dimension change is handled internally
    by the MLP, so no separate :class:`ResampleModule` is needed.

    Parameters
    ----------
    t_size : int
        Number of output timesteps.
    c_size : int
        Number of output channels.
    t_h_size : int
        Number of latent timesteps.
    c_h_size : int
        Number of latent channels.
    hps : dict
        Hyper-parameters forwarded to :class:`SubMLP`.
    """

    def __init__(self, t_size: int, c_size: int, t_h_size: int, c_h_size: int, hps: dict) -> None:
        super().__init__()
        self.mlp = SubMLP(in_dim=[t_h_size, c_h_size], out_dim=[t_size, c_size], hps=hps)

    def forward(self, x: Tensor) -> Tensor:
        """Decode the latent ``x`` to the output shape.

        Parameters
        ----------
        x : Tensor
            Latent of shape ``[B, t_h_size, c_h_size]``.

        Returns
        -------
        Tensor
            Reconstruction of shape ``[B, t_size, c_size]``.
        """
        return self.mlp(x)


# -----------
# SUB-MODULES
# -----------

class SubLSTM(KnowItLSTM):
    """Thin wrapper around :class:`KnowItLSTM` with AE-friendly defaults.

    Parameters
    ----------
    in_dim : int
        Input channel dimension.
    out_dim : int
        Output channel dimension.
    hps : dict
        Accepted keys: ``'depth'`` (int, default 1), ``'dropout'`` (float,
        default 0.0), ``'width'`` (int, default 256), ``'hc_init_method'``
        (str, default ``'zeros'``), ``'layernorm'`` (bool, default ``True``),
        ``'bidirectional'`` (bool, default ``False``).
    """

    def __init__(self, in_dim: int, out_dim: int, hps: dict) -> None:
        super().__init__(input_dim=[1, in_dim],
                         output_dim=[1, out_dim],
                         task_name='vl_regression',
                         stateful=False,
                         depth=hps['depth'] if 'depth' in hps else 1,
                         dropout=hps['dropout'] if 'dropout' in hps else 0.0,
                         width=hps['width'] if 'width' in hps else 256,
                         hc_init_method=hps['hc_init_method'] if 'hc_init_method' in hps else 'zeros',
                         layernorm=hps['layernorm'] if 'layernorm' in hps else True,
                         bidirectional=hps['bidirectional'] if 'bidirectional' in hps else False,
                         residual=False)


class SubCNN(KnowItCNN):
    """Thin wrapper around :class:`KnowItCNN` with AE-friendly defaults.

    Parameters
    ----------
    in_dim : int
        Input channel dimension.
    out_dim : int
        Output channel dimension.
    hps : dict
        Accepted keys: ``'depth'`` (int, default 3), ``'num_filters'`` (int,
        default 64), ``'kernel_size'`` (int, default 3), ``'normalization'``
        (str, default ``'batch'``), ``'dropout'`` (float, default 0.0),
        ``'residual_connect'`` (bool, default ``True``), ``'dilation_base'``
        (int, default 2).
    """

    def __init__(self, in_dim: int, out_dim: int, hps: dict) -> None:
        super().__init__(input_dim=[1, in_dim],
                         output_dim=[1, out_dim],
                         task_name='vl_regression',
                         depth=hps['depth'] if 'depth' in hps else 3,
                         num_filters=hps['num_filters'] if 'num_filters' in hps else 64,
                         kernel_size=hps['kernel_size'] if 'kernel_size' in hps else 3,
                         normalization=hps['normalization'] if 'normalization' in hps else 'batch',
                         dropout=hps['dropout'] if 'dropout' in hps else 0.0,
                         residual_connect=hps['residual_connect'] if 'residual_connect' in hps else True,
                         dilation_base=hps['dilation_base'] if 'dilation_base' in hps else 2,
                         )


class SubMLP(KnowItMLP):
    """Thin wrapper around :class:`KnowItMLP` with AE-friendly defaults.

    Parameters
    ----------
    in_dim : list[int]
        ``[T, C]`` shape of the input (flattened to ``T * C`` features).
    out_dim : list[int]
        ``[T, C]`` shape of the output (MLP output is reshaped to this).
    hps : dict
        Accepted keys: ``'depth'`` (int, default 3), ``'width'`` (int,
        default 256), ``'dropout'`` (float, default 0.0), ``'activations'``
        (str, default ``'ReLU'``), ``'output_activation'`` (str | None,
        default ``None``), ``'batchnorm'`` (bool, default ``False``).
    """

    def __init__(self, in_dim: list[int], out_dim: list[int], hps: dict) -> None:
        super().__init__(input_dim=in_dim,
                         output_dim=out_dim,
                         task_name='regression',
                         depth=hps['depth'] if 'depth' in hps else 3,
                         width=hps['width'] if 'width' in hps else 256,
                         dropout=hps['dropout'] if 'dropout' in hps else 0.0,
                         activations=hps['activations'] if 'activations' in hps else 'ReLU',
                         output_activation=hps['output_activation'] if 'output_activation' in hps else None,
                         batchnorm=hps['batchnorm'] if 'batchnorm' in hps else False,
                         )

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.contiguous())


class ResampleModule(nn.Module):
    """Single-layer conv or transposed-conv that changes the time dimension.

    Maps ``[B, from_size, channels]`` → ``[B, to_size, channels]`` using a
    kernel whose size exactly spans the difference between the two lengths.

    Parameters
    ----------
    channels : int
        Number of channels (same for input and output).
    from_size : int
        Input time dimension.
    to_size : int
        Output time dimension.
    """

    def __init__(self, channels: int, from_size: int, to_size: int) -> None:
        super().__init__()

        if to_size >= from_size:
            self.conv = nn.ConvTranspose1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=to_size - from_size + 1,
                stride=1,
                padding=0
            )
        else:
            self.conv = nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=from_size - to_size + 1,
                stride=1,
                padding=0
            )

    def forward(self, x: Tensor) -> Tensor:
        """Resample ``x`` from ``from_size`` to ``to_size`` timesteps.

        Parameters
        ----------
        x : Tensor
            Input of shape ``[B, from_size, channels]``.

        Returns
        -------
        Tensor
            Output of shape ``[B, to_size, channels]``.
        """
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        return x


class SphericalNorm(nn.Module):
    """RMS-normalise a tensor so it lies on the unit hypersphere.

    Flattens all non-batch dimensions, divides by the per-sample RMS, and
    returns a flat ``[B, N]`` tensor.  Use ``.view()`` on the result to
    restore the original shape.

    Parameters
    ----------
    eps : float
        Small constant added inside the square root for numerical stability.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, z: Tensor) -> Tensor:
        """Normalise ``z`` by its per-sample RMS across all non-batch dimensions.

        Parameters
        ----------
        z : Tensor
            Input of arbitrary shape ``[B, ...]``.

        Returns
        -------
        Tensor
            Flat normalised tensor of shape ``[B, N]`` where
            ``N = product of all non-batch dimensions``.
        """
        x = z.flatten(1)
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms


class Conditioner(nn.Module):
    """Applies class conditioning to a ``[B, T, C]`` tensor before encoding or decoding.

    Supports four modes selected by ``conditioning_mode``:

    - ``'class_concatenate'``: broadcasts a learned class embedding over the
      time axis and appends it as extra channels, increasing ``C`` by
      ``class_dim``.
    - ``'class_film'``: applies FiLM modulation ``(1 + γ) * x + β`` where
      ``(γ, β)`` are projected from a learned class embedding.  Output shape
      is unchanged.  Projection weights are zero-initialised so the identity
      is the starting point.
    - ``{'class_aux_loss': {'beta': float}}``: the forward pass is a no-op
      (input is returned unchanged), but the input is cached so that
      :meth:`compute_aux_loss` can classify it with a linear head and return
      a weighted cross-entropy loss.
    - ``None``: identity — the input is returned unchanged.

    Parameters
    ----------
    t_dim : int
        Number of timesteps in the input tensor.
    c_dim : int
        Number of channels in the input tensor.
    num_classes : int | None
        Number of distinct classes.  Required for all non-``None`` modes.
    class_dim : int
        Embedding dimension used by ``'class_concatenate'`` and ``'class_film'``.
    conditioning_mode : str | dict | None
        Conditioning mode (see above).  Pass a dict only for
        ``'class_aux_loss'``: ``{'class_aux_loss': {'beta': 1.0}}``.
    """

    def __init__(self, t_dim: int, c_dim: int, num_classes: int, class_dim: int, conditioning_mode: str | dict | None = None) -> None:
        super().__init__()

        self.t_dim = t_dim
        self.c_dim = c_dim

        if isinstance(conditioning_mode, dict):
            self.conditioning_mode = next(iter(conditioning_mode))
            self.conditioning_hps = conditioning_mode[self.conditioning_mode]
        else:
            self.conditioning_mode = conditioning_mode
            self.conditioning_hps = {}

        if self.conditioning_mode not in ('class_concatenate', 'class_film', 'class_aux_loss', None):
            logger.error(f'Unknown conditioning_mode: {self.conditioning_mode}')
            exit(101)

        if self.conditioning_mode in ('class_concatenate', 'class_film'):
            self.num_classes = num_classes
            self.class_dim = class_dim
            self.class_embedding = self._prepare_class_embeddings()

        if self.conditioning_mode == 'class_film':
            self.film_proj = nn.Linear(class_dim, 2 * c_dim)
            nn.init.zeros_(self.film_proj.weight)
            nn.init.zeros_(self.film_proj.bias)

        if self.conditioning_mode == 'class_aux_loss':
            if num_classes is None:
                logger.error('class_aux_loss requires num_classes.')
                exit(101)
            self.num_classes = num_classes
            self.cls_head = nn.Linear(t_dim * c_dim, num_classes)
            self._cached_latent = None

        self.t_dim_out = t_dim
        self.c_dim_out = c_dim
        if self.conditioning_mode == 'class_concatenate':
            self.c_dim_out = c_dim + self.class_dim

    def forward(self, x: Tensor, y: Tensor | None) -> Tensor:
        """Apply class conditioning to ``x`` and return the (possibly modified) tensor.

        Parameters
        ----------
        x : Tensor
            Input of shape ``[B, T, C]``.
        y : Tensor | None
            Class labels (integer indices ``[B]`` or one-hot ``[B, num_classes]``).
            Pass ``None`` for unconditional inference; a zero embedding is used in
            that case for ``class_concatenate`` and ``class_film`` modes.

        Returns
        -------
        Tensor
            Conditioned tensor.  Shape is ``[B, T, C + class_dim]`` for
            ``'class_concatenate'``; ``[B, T, C]`` for all other modes.
        """
        if self.conditioning_mode == 'class_concatenate':
            batch_size = x.shape[0]
            class_emb = self._get_class_embedding(y, batch_size, x.device)
            if class_emb is not None:
                cond_map = class_emb.unsqueeze(1).expand(class_emb.shape[0], x.shape[1], class_emb.shape[1])
                x = torch.cat([x, cond_map], dim=-1)
        elif self.conditioning_mode == 'class_film':
            batch_size = x.shape[0]
            class_emb = self._get_class_embedding(y, batch_size, x.device)
            if class_emb is not None:
                film_params = self.film_proj(class_emb)          # [B, 2 * c_dim]
                gamma, beta = film_params.chunk(2, dim=-1)        # each [B, c_dim]
                gamma = gamma.unsqueeze(1)                        # [B, 1, c_dim] → broadcasts over T
                beta = beta.unsqueeze(1)
                x = (1 + gamma) * x + beta
        elif self.conditioning_mode == 'class_aux_loss':
            self._cached_latent = x

        return x

    def compute_aux_loss(self, y: Tensor) -> Tensor:
        x = self._cached_latent
        if x.dim() == 3:
            x = x.reshape(x.shape[0], -1)
        logits = self.cls_head(x)
        labels = torch.argmax(y, dim=-1) if y.dim() > 1 else y.long()
        return F.cross_entropy(logits, labels)

    def get_beta_cls(self) -> float:
        return self.conditioning_hps.get('beta', 1.0)

    def get_out_dim(self) -> tuple:
        return self.t_dim_out, self.c_dim_out

    def _prepare_class_embeddings(self) -> nn.Embedding:
        """Initialise the class embedding table.

        Sets ``self.class_embedding`` to an ``nn.Embedding`` of size
        ``(num_classes, class_dim)``.  Called automatically from ``__init__``
        when ``conditioning_mode`` is ``'class_concatenate'`` or ``'class_film'``.
        """
        if self.num_classes is None:
            logger.error('num_classes is None. Cannot generate class embeddings.')
            exit(101)
        return nn.Embedding(self.num_classes, self.class_dim)

    def _get_class_embedding(self, y: Tensor | None, batch_size: int, device: torch.device) -> Tensor | None:
        """Look up class embeddings for a batch.

        Returns ``None`` when no embedding table is present, or a zero vector
        when ``y`` is ``None`` (unconditional generation at inference time).

        Parameters
        ----------
        y : Tensor | None
            Integer class indices ``[B]`` or one-hot ``[B, num_classes]``.
            Pass ``None`` to obtain a zero embedding (unconditional).
        batch_size : int
            Expected batch size.
        device : torch.device
            Device on which to create zero-embedding tensors.

        Returns
        -------
        Tensor | None
            Class embedding of shape ``[B, class_dim]``, or ``None`` if
            ``self.class_embedding`` is not initialised.
        """
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


# --------------------
# EXPERIMENTAL MODULES
# --------------------

class ConvBlock2D(nn.Module):
    """2D variant of :class:`CNN.ConvBlock`.

    Applies a ``Conv2d`` with optional normalisation, activation, dropout, and
    residual connection.  Input and output tensors have shape
    ``[B, C, H, W]`` (standard PyTorch 2D channel-first format).

    Parameters
    ----------
    n_inputs : int
        Number of input channels.
    n_outputs : int
        Number of output channels.
    kernel_size : int
        Square kernel size (applied identically in H and W).
    dilation : int
        Dilation factor (applied identically in H and W).
    padding : int
        Padding (applied identically in H and W).  For ``kernel_size=3`` set
        ``padding=dilation`` to preserve spatial dimensions.
    dropout : float
        Dropout probability (0 disables dropout).
    normalization : str | None
        ``'batch'`` for :class:`~torch.nn.BatchNorm2d`, ``'weight'`` for
        weight normalisation, or ``None`` for no normalisation.
    activations : str
        Name of an ``nn.`` activation class (e.g. ``'ReLU'``).
    residual_connect : bool
        If ``True``, adds a skip connection; a 1×1 ``Conv2d`` is used when
        ``n_inputs != n_outputs``.
    """

    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int,
                 dilation: int, padding: int, dropout: float,
                 normalization: str | None, activations: str,
                 residual_connect: bool) -> None:
        super().__init__()

        layers = [nn.Conv2d(n_inputs, n_outputs, kernel_size,
                            stride=1, padding=padding, dilation=dilation)]
        if normalization == 'weight':
            layers[0] = weight_norm(layers[0])
        elif normalization == 'batch':
            layers.append(nn.BatchNorm2d(n_outputs))
        elif normalization is None:
            pass
        else:
            logger.error('Unknown normalization type %s.', normalization)
            exit(101)
        layers.append(getattr(nn, activations)())
        if dropout:
            layers.append(nn.Dropout(p=dropout))
        self.block = nn.Sequential(*layers)
        init_mod(self.block)

        self.res_connect = None
        if residual_connect:
            if n_inputs != n_outputs:
                self.res_connect = nn.Conv2d(n_inputs, n_outputs, kernel_size=1)
                init_mod(self.res_connect)
            else:
                self.res_connect = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Apply the conv block with optional residual connection.

        Parameters
        ----------
        x : Tensor
            Input of shape ``[B, n_inputs, H, W]``.

        Returns
        -------
        Tensor
            Output of shape ``[B, n_outputs, H, W]``.
        """
        out = self.block(x)
        if self.res_connect is not None:
            out = out + self.res_connect(x)
        return out


class SubCNN2D(nn.Module):
    """2D CNN sub-module that maps ``[B, T, C_in]`` → ``[B, T, C_out]``.

    Applies 2D convolutions over the joint ``(T, C)`` plane, capturing spatial
    structure that 1D convolutions along the time axis alone cannot express.
    Intended as a drop-in replacement for :class:`SubCNN` when the input is
    better treated as a 2D image (e.g. 28×28 MNIST digits stored as
    ``[B, T=28, C=28]``).

    Internally the input is reshaped to a single-channel 2D image
    ``[B, 1, T, C_in]``.  A stack of :class:`ConvBlock2D` layers with
    same-size padding preserves both spatial dimensions.  A final ``Conv2d``
    with kernel ``(1, C_in)`` collapses the component dimension, producing
    ``[B, C_out, T, 1]``, which is reshaped to ``[B, T, C_out]``.

    Parameters
    ----------
    in_dim : int
        Input component dimension (``C_in``).
    out_dim : int
        Output component dimension (``C_out``).
    hps : dict
        Accepted keys (all optional):

        - ``'depth'`` (int, default 3): number of :class:`ConvBlock2D` layers.
        - ``'num_filters'`` (int, default 64): feature maps in hidden layers.
        - ``'kernel_size'`` (int, default 3): square kernel size.
        - ``'normalization'`` (str | None, default ``'batch'``).
        - ``'dropout'`` (float, default 0.0).
        - ``'residual_connect'`` (bool, default ``True``).
        - ``'dilation_base'`` (int, default 2): dilation at block ``i`` is
          ``dilation_base ** i``.
        - ``'activations'`` (str, default ``'ReLU'``).
    """

    def __init__(self, in_dim: int, out_dim: int, hps: dict) -> None:
        super().__init__()

        self.in_dim = in_dim
        depth = hps.get('depth', 3)
        num_filters = hps.get('num_filters', 64)
        kernel_size = hps.get('kernel_size', 3)
        normalization = hps.get('normalization', 'batch')
        dropout = hps.get('dropout', 0.0)
        residual_connect = hps.get('residual_connect', True)
        dilation_base = hps.get('dilation_base', 2)
        activations = hps.get('activations', 'ReLU')

        blocks = []
        for i in range(depth):
            n_in = 1 if i == 0 else num_filters
            dilation = dilation_base ** i
            blocks.append(ConvBlock2D(
                n_inputs=n_in,
                n_outputs=num_filters,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=dilation,
                dropout=dropout,
                normalization=normalization,
                activations=activations,
                residual_connect=residual_connect,
            ))
        self.conv_blocks = nn.Sequential(*blocks)

        # collapse C dimension: [B, num_filters, T, C_in] → [B, out_dim, T, 1]
        final_in = num_filters if depth > 0 else 1
        self.final_conv = nn.Conv2d(final_in, out_dim, kernel_size=(1, in_dim))
        init_mod(self.final_conv)

    def forward(self, x: Tensor) -> Tensor:
        """Map ``x`` from ``[B, T, C_in]`` to ``[B, T, C_out]`` via 2D convolutions.

        Parameters
        ----------
        x : Tensor
            Input of shape ``[B, T, C_in]``.

        Returns
        -------
        Tensor
            Output of shape ``[B, T, C_out]``.
        """
        x = x.unsqueeze(1)           # [B, T, C_in] → [B, 1, T, C_in]
        x = self.conv_blocks(x)      # → [B, num_filters, T, C_in]
        x = self.final_conv(x)       # → [B, out_dim, T, 1]
        x = x.squeeze(-1)            # → [B, out_dim, T]
        x = x.permute(0, 2, 1)       # → [B, T, out_dim]
        return x


class CNN2DEncoder(nn.Module):
    """2D CNN encoder that maps ``[B, t_size, c_size]`` → ``[B, t_h_size, c_h_size]``.

    Uses :class:`SubCNN2D` to apply 2D convolutions over the joint ``(T, C)``
    plane.  A :class:`ResampleModule` handles any mismatch between ``t_size``
    and ``t_h_size``.

    Parameters
    ----------
    t_size : int
        Number of input timesteps.
    c_size : int
        Number of input channels.
    t_h_size : int
        Number of latent timesteps.
    c_h_size : int
        Number of latent channels.
    hps : dict
        Hyper-parameters forwarded to :class:`SubCNN2D`.
    """

    def __init__(self, t_size: int, c_size: int, t_h_size: int, c_h_size: int, hps: dict) -> None:
        super().__init__()

        self.t_size = t_size
        self.c_size = c_size
        self.t_h_size = t_h_size
        self.c_h_size = c_h_size

        self.cnn = SubCNN2D(in_dim=self.c_size, out_dim=self.c_h_size, hps=hps)

        if self.t_h_size != self.t_size:
            self.resampler = ResampleModule(self.c_h_size, self.t_size, self.t_h_size)

    def forward(self, x: Tensor) -> Tensor:
        """Encode ``x`` to the latent time resolution.

        Parameters
        ----------
        x : Tensor
            Input of shape ``[B, t_size, c_size]``.

        Returns
        -------
        Tensor
            Latent of shape ``[B, t_h_size, c_h_size]``.
        """
        z = self.cnn(x)

        if self.t_h_size != self.t_size:
            z = self.resampler(z)

        return z


class CNN2DDecoder(nn.Module):
    """2D CNN decoder that maps ``[B, t_h_size, c_h_size]`` → ``[B, t_size, c_size]``.

    Uses :class:`SubCNN2D` to apply 2D convolutions over the joint ``(T, C)``
    plane.  A :class:`ResampleModule` handles any mismatch between ``t_h_size``
    and ``t_size``.

    Parameters
    ----------
    t_size : int
        Number of output timesteps.
    c_size : int
        Number of output channels.
    t_h_size : int
        Number of latent timesteps.
    c_h_size : int
        Number of latent channels.
    hps : dict
        Hyper-parameters forwarded to :class:`SubCNN2D`.
    """

    def __init__(self, t_size: int, c_size: int, t_h_size: int, c_h_size: int, hps: dict) -> None:
        super().__init__()

        self.t_size = t_size
        self.c_size = c_size
        self.t_h_size = t_h_size
        self.c_h_size = c_h_size

        self.cnn = SubCNN2D(in_dim=self.c_h_size, out_dim=self.c_size, hps=hps)

        if self.t_h_size != self.t_size:
            self.resampler = ResampleModule(self.c_size, self.t_h_size, self.t_size)

    def forward(self, x: Tensor) -> Tensor:
        """Decode the latent ``x`` to the output time resolution.

        Parameters
        ----------
        x : Tensor
            Latent of shape ``[B, t_h_size, c_h_size]``.

        Returns
        -------
        Tensor
            Reconstruction of shape ``[B, t_size, c_size]``.
        """
        z = self.cnn(x)

        if self.t_h_size != self.t_size:
            z = self.resampler(z)

        return z
