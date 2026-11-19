# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path

import torch
from diffusers import AutoencoderOobleck

# Set env vars for local T5 loading
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from inference.utils import print_rank_0


class _DiffusersAudioVAEAdapter:
    """Keep the project's tensor-only VAE interface over Diffusers outputs."""

    def __init__(self, vae):
        self.vae = vae

    def to(self, *args, **kwargs):
        self.vae.to(*args, **kwargs)
        return self

    def encode(self, audio):
        squeeze_batch = audio.ndim == 2
        if squeeze_batch:
            audio = audio.unsqueeze(0)
        latents = self.vae.encode(audio).latent_dist.mode()
        return latents.squeeze(0) if squeeze_batch else latents

    def decode(self, latents):
        squeeze_batch = latents.ndim == 2
        if squeeze_batch:
            latents = latents.unsqueeze(0)
        waveform = self.vae.decode(latents).sample
        return waveform.squeeze(0) if squeeze_batch else waveform


class SAAudioFeatureExtractor:
    """Stable Audio Feature Extractor that loads model once and reuses it."""

    def __init__(self, device, model_path):
        """Initialize the extractor with model loading."""
        self.device = device
        self.vae_model, self.sample_rate = self._get_vae_only(model_path)
        # self.vae_model.to(self.device).to(torch.bfloat16)
        self.resampler = None  # Will be initialized when needed

    def _get_vae_only(self, model_path):
        """Load only the local Diffusers Stable Audio Oobleck VAE."""
        if isinstance(model_path, str) and Path(model_path).is_dir():
            try:
                vae_path = Path(model_path) / "vae"
                if not (vae_path / "config.json").exists():
                    raise FileNotFoundError(f"VAE config does not exist: {vae_path / 'config.json'}")
                if not (vae_path / "diffusion_pytorch_model.safetensors").exists():
                    raise FileNotFoundError(
                        f"VAE weights do not exist: {vae_path / 'diffusion_pytorch_model.safetensors'}"
                    )

                vae = AutoencoderOobleck.from_pretrained(
                    model_path,
                    subfolder="vae",
                    torch_dtype=torch.float32,
                    local_files_only=True,
                )
                vae = _DiffusersAudioVAEAdapter(vae).to(self.device)
                return vae, vae.vae.config.sampling_rate

            except Exception as e:
                print_rank_0(f"audio model loading failed: {e}")
                raise RuntimeError(
                    "Failed to load VAE-only Stable Audio model from local path"
                ) from e
        else:
            print_rank_0("Non-local path is not supported in audio model loading")

    def decode(self, latents):
        with torch.no_grad():
            waveform_out = self.vae_model.decode(latents)
        return waveform_out

    def encode(self, waveform):
        with torch.no_grad():
            latents = self.vae_model.encode(waveform)
        return latents
