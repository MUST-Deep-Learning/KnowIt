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
from default_archs.CNN import Model as KnowItCNN
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
    **decode**.  The first and last stages can optionally concatenate a learned class embedding
    to the input tensor when ``enc_is_class_conditional=True`` or ``dec_is_class_conditional=True``.

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
        Backbone for the encoder: ``'LSTM'`` (:class:`LSTMEncoder`) or
        ``'CNN'`` (:class:`CNNEncoder`).
    decoder_type : str
        Backbone for the decoder: ``'LSTM'`` (:class:`LSTMDecoder`) or
        ``'CNN'`` (:class:`CNNDecoder`).
    enc_is_class_conditional : bool
        If ``True``, a learned class embedding is concatenated to the encoder
        input, allowing the encoder to exploit the class label.  When
        ``False`` the latent must encode class-discriminative information
        itself, which naturally separates classes in latent space.
    dec_is_class_conditional : bool
        If ``True``, a learned class embedding is concatenated to the latent
        before decoding, giving the decoder explicit access to the class label.
    num_classes : int | None
        Number of distinct classes.  Required when ``enc_is_class_conditional=True`` or ``dec_is_class_conditional=True``.
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
        enc_is_class_conditional: bool = False,
        dec_is_class_conditional: bool = False,
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
        if (enc_is_class_conditional or dec_is_class_conditional) and num_classes is None:
            logger.error('No num_classes specified for class_conditional.')
            exit(101)
        self.enc_is_class_conditional = enc_is_class_conditional
        self.dec_is_class_conditional = dec_is_class_conditional
        self.num_classes = num_classes
        self.class_dim = class_dim
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.manifold_type = manifold_type
        self.rec_loss_f = _parse_rec_loss(rec_loss)
        self.class_embedding: nn.Embedding | None = None

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

        if self.enc_is_class_conditional or self.dec_is_class_conditional:
            self.prepare_class_embeddings()

        enc_in = self.c_size + self.class_dim if self.enc_is_class_conditional else self.c_size
        dec_in = self.c_h_size + self.class_dim if self.dec_is_class_conditional else self.c_h_size
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
            ``enc_is_class_conditional=True`` or ``dec_is_class_conditional=True``.

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
        """Optionally prepend class embeddings and run the encoder.

        Parameters
        ----------
        x : Tensor
            Input of shape ``[B, T, C]``.
        y : Tensor | None
            Class labels; forwarded to :meth:`concat_class_embeddings`.

        Returns
        -------
        Tensor
            Latent representation of shape ``[B, t_h_size, c_h_size]``.
        """
        if self.enc_is_class_conditional:
            x = self.concat_class_embeddings(x, y)
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
        """Optionally prepend class embeddings and run the decoder.

        Parameters
        ----------
        z : Tensor
            Latent tensor of shape ``[B, t_h_size, c_h_size]``.
        y : Tensor | None
            Class labels; forwarded to :meth:`concat_class_embeddings`.

        Returns
        -------
        Tensor
            Reconstruction of shape ``[B, T, C]``.
        """
        if self.dec_is_class_conditional:
            z = self.concat_class_embeddings(z, y)
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

    # CLASS CONDITIONING FUNCTIONS

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

    def prepare_class_embeddings(self) -> None:
        """Initialise the class embedding table.

        Sets ``self.class_embedding`` to an ``nn.Embedding`` of size
        ``(num_classes, class_dim)``.  Called automatically from ``__init__``
        when ``enc_is_class_conditional=True`` or ``dec_is_class_conditional=True``.
        """
        if self.num_classes is None:
            logger.error('num_classes is None. Cannot generate class embeddings.')
            exit(101)
        self.class_embedding = nn.Embedding(self.num_classes, self.class_dim)

    def concat_class_embeddings(self, x: Tensor, y: Tensor | None) -> Tensor:
        """Append class embedding channels to ``x`` along the component axis.

        Parameters
        ----------
        x : Tensor
            Input of shape ``[B, T, C]``.
        y : Tensor | None
            Class labels; ``None`` yields a zero embedding.

        Returns
        -------
        Tensor
            Concatenated tensor of shape ``[B, T, C + class_dim]`` when class
            conditioning is active, otherwise the original ``[B, T, C]``.
        """
        batch_size = x.shape[0]
        class_emb = self._get_class_embedding(y, batch_size, x.device)
        if class_emb is not None:
            cond_map = class_emb.unsqueeze(1).expand(class_emb.shape[0], x.shape[1], class_emb.shape[1])
            x = torch.cat([x, cond_map], dim=-1)
        return x

    # GENERATING FUNCTIONS

    def sample_by_class(self, n_samples: int, class_id: int | list[int]) -> Tensor:
        """Sample from the latent prior conditioned on a specific class.

        Samples ``z ~ N(0, I)`` in latent space, projects through the manifold,
        and decodes with the requested class conditioning.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        class_id : int | list[int]
            A single integer (all samples from the same class) or a list of
            integers of length ``n_samples`` (per-sample class assignment).

        Returns
        -------
        Tensor
            Generated time series of shape ``[n_samples, T, C]``.
        """
        if not self.dec_is_class_conditional:
            logger.error('dec_is_class_conditional is False. Cannot sample by class.')
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

    def sample(self, n_samples: int) -> Tensor:
        """Sample from the latent prior without class conditioning.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.

        Returns
        -------
        Tensor
            Generated time series of shape ``[n_samples, T, C]``.
        """
        device = next(self.parameters()).device
        z = torch.normal(size=(n_samples, self.t_h_size, self.c_h_size), mean=0, std=1., device=device)

        z = self.manifold_proj(z)
        # z = self.manifold.spherifier(z)
        # z = z.view(-1, self.t_h_size, self.c_h_size)

        return self.decode(z)

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
            if self.dec_is_class_conditional:
                v_NOISY = self.concat_class_embeddings(v_NOISY, y)
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

        - ``'kl_warmup_epochs'`` (int, default 0): number of epochs over
          which to linearly ramp ``beta_kl`` from 0 to its target value.
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
        logvar = self.to_logvar(z)

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
        """Linearly warm ``beta_kl`` from 0 to its target over ``kl_warmup_epochs``.

        During validation / test (``training=False``) the full ``beta_kl`` is
        applied so that logged metrics are comparable across runs.

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
            frac = min(1.0, float(epoch + 1) / float(self.kl_warmup_epochs))
            self.current_beta_kl = min(self.beta_kl, self.beta_kl * frac)
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

    def forward(self, z: Tensor) -> Tensor:
        """Project ``z`` onto the sphere and generate perturbed latents.

        Stores the clean sphere projection ``self.v`` (shape ``[B, T*C]``)
        and the large-noise projection ``self.v_NOISY`` (shape
        ``[B, t_h_size, c_h_size]``) for use in the consistency losses.

        Parameters
        ----------
        z : Tensor
            Encoder output of shape ``[B, t_h_size, c_h_size]``.

        Returns
        -------
        Tensor
            Small-noise sphere projection of shape ``[B, t_h_size, c_h_size]``.
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