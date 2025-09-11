from abc import ABC, abstractmethod

import torch

from omegaconf import DictConfig
from scipy.signal import resample
from speechtokenizer import SpeechTokenizer

from BigCodec.vq.codec_decoder import CodecDecoder
from BigCodec.vq.codec_encoder import CodecEncoder
from WavTokenizer.decoder.pretrained import WavTokenizer
from salt.tokenization.utils import get_audio_start_end_tokens


class AudioTokenizer(ABC):
    """
    Abstract base class for audio tokenizer implementations. All audio tokenizers should implement its interface.
    """

    def __init__(self, config: DictConfig):
        self.config = config

        self.start_audio_token_id = config.start_audio_token_id
        self.end_audio_token_id = config.end_audio_token_id
        self.n_text_tokens = config.n_text_tokens

        self.sample_rate = config.sample_rate

        self.quantizer = None
        self.device = config.device

    def encode(self, audio_dict: dict) -> torch.Tensor:
        """
        Encodes audio to tokens. Calls _encode method, which is specific for each audio tokenizer.

        :param audio_dict: dictionary containing audio array (numpy) and sample rate.
        :return: codes with shape (n_codebooks, compressed_audio_length).
        """
        audio = audio_dict["array"]
        sr = audio_dict["sample_rate"]

        if sr != self.sample_rate:
            audio = resample(audio, audio.shape[-1] * self.sample_rate // sr)

        audio = torch.from_numpy(audio).float().to(self.device)

        return self._encode(audio.unsqueeze(0))

    @abstractmethod
    def _encode(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Encodes audio using tokenizer specific logic. This method must be implemented by subclasses.

        :param audio: 1-D torch tensor.
        :return: codes with shape (n_codebooks, compressed_audio_length).
        """
        pass

    @abstractmethod
    def _decode(self, audio_tokens: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs audio from tokens. This method must be implemented by subclasses.
        Takes flattened codes as input.

        :param audio_tokens: tensor with shape (1, n_codebooks * compressed_audio_length).
        :return: reconstructed audio, shape (1, audio_length).
        """
        pass

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs audio from tokens. Calls _decode method, which is specific for each audio tokenizer.
        If num_codebooks > 1, flattening must be done properly.

        Example with 3 codebooks:
        codes = tensor(
            [
                [1, 2],
                [3, 4],
                [5, 6],
            ]
        )
        must be reshaped to
        codes = tensor([[1, 3, 5, 2, 4, 6]])

        :param tokens: tensor with shape (1, n_codebooks * compressed_audio_length).
        :return: reconstructed audio, shape (1, audio_length).
        """
        start, end = get_audio_start_end_tokens(
            tokens, self.start_audio_token_id, self.end_audio_token_id
        )
        audio_tokens = tokens[:, start:end]
        audio_tokens = audio_tokens % self.n_text_tokens
        return self._decode(audio_tokens)


class SpeechTokenizerWrapper(AudioTokenizer):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.quantizer = SpeechTokenizer.load_from_checkpoint(
            config.quantizer_config_path, config.quantizer_ckpt_path
        )
        self.quantizer = self.quantizer.to(self.device)
        self.n_codebooks = config.n_codebooks

    def _encode(self, audio: torch.Tensor) -> torch.Tensor:
        audio = audio.unsqueeze(0)
        codes = self.quantizer.encode(audio)
        return codes.squeeze(1)

    def _decode(self, audio_tokens: torch.Tensor) -> torch.Tensor:
        remainder = audio_tokens.shape[-1] % self.n_codebooks

        if remainder:
            # pad if last frame is incomplete
            # zero padding is used now, for speechtokenizer using get_audio_padding_tokens is also possible
            pad_tokens = torch.zeros(
                1, self.n_codebooks - remainder, device=self.device, dtype=torch.long
            )
            audio_tokens = torch.cat([audio_tokens, pad_tokens], dim=1)

        transposed = audio_tokens.view(-1, self.n_codebooks).t()
        codes = transposed.view(self.n_codebooks, 1, -1).to(self.device)

        return self.quantizer.decode(codes).squeeze(0)


class WavTokenizerWrapper(AudioTokenizer):
    def __init__(self, config: DictConfig):
        super().__init__(config)

        self.quantizer = WavTokenizer.from_pretrained0802(
            config.quantizer_config_path, config.quantizer_ckpt_path
        )
        self.quantizer = self.quantizer.to(self.device)

    def _encode(self, audio: torch.Tensor) -> torch.Tensor:
        bandwidth_id = torch.tensor([0])

        audio = audio.to(self.device)
        _, codes = self.quantizer.encode_infer(audio, bandwidth_id=bandwidth_id)
        return codes.squeeze(1)

    def _decode(self, audio_tokens: torch.Tensor) -> torch.Tensor:
        transposed = audio_tokens.view(-1, 1).t()
        codes = transposed.view(1, 1, -1).to(self.device)

        features = self.quantizer.codes_to_features(codes)
        bandwidth_id = torch.tensor([0], device=self.device)

        return self.quantizer.decode(features, bandwidth_id=bandwidth_id).squeeze(0)


class BigCodecWrapper(AudioTokenizer):
    def __init__(self, config: DictConfig):
        super().__init__(config)

        ckpt = torch.load(config.quantizer_ckpt_path, map_location="cpu")
        encoder = CodecEncoder()
        encoder.load_state_dict(ckpt["CodecEnc"])
        self.encoder = encoder.eval().to(self.device)

        decoder = CodecDecoder()
        decoder.load_state_dict(ckpt["generator"])
        self.decoder = decoder.eval().to(self.device)

    def _encode(self, audio: torch.Tensor) -> torch.Tensor:
        vq_emb = self.encoder(audio.unsqueeze(1))
        _, vq_code, _ = self.decoder(vq_emb, vq=True)
        return vq_code

    def _decode(self, audio_tokens: torch.Tensor) -> torch.Tensor:
        audio_tokens = audio_tokens.reshape(1, -1, 1).to(self.device)
        emb = self.decoder.vq2emb(audio_tokens).transpose(1, 2)
        return self.decoder(emb, vq=False).squeeze()
