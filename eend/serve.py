#!/usr/bin/env python3
# author: Linxiao ZENG

import os
import time
import json
import yamlargparse
from typing import Iterable, List

import numpy as np
from scipy.signal import medfilt
import torch

from eend import feature
from eend.kaldi_data import load_wav
from eend.pytorch_backend.infer import (
    clustering,
    get_cl_sil,
    merge_acti_clslab,
    prepare_model_for_eval,
    _gen_chunk_indices,
    stitching
)


class DiarizationServeModel:
    """Diarization model ready to be served on audio input."""

    def __init__(
        self,
        eend_model,
        frame_size,
        frame_shift,
        input_transform,
        context_size,
        subsampling,
        chunk_size,
        sil_spk_th=0.05,
        ahc_dis_th=1.0,
        clink_dis=1e+4,
    ) -> None:
        """Initialize EEND model with necessary options."""
        self._model = eend_model
        self.num_speakers = eend_model.n_speakers
        self.chunk_size = chunk_size
        # audio process related
        self.frame_size = frame_size
        self.frame_shift = frame_shift
        self.input_transform = input_transform
        self.context_size = context_size
        self.subsampling = subsampling
        # Clustering related
        self.sil_spk_th = sil_spk_th
        self.ahc_dis_th = ahc_dis_th
        self.clink_dis = clink_dis

    @staticmethod
    def add_args(parser) -> None:
        """Add arguments to the parser."""
        audio_args = parser.add_argument_group('Audio processing config')
        audio_args.add_argument(
            '--input-transform', default='',
            choices=[
                '', 'log', 'logmel', 'logmel23', 'logmel23_swn', 'logmel23_mn'
            ],
            help='input transform',
        )
        audio_args.add_argument(
            '--chunk-size', default=2000, type=int,
            help='input is chunked with this size')
        audio_args.add_argument(
            '--context-size', default=0, type=int,
            help='frame splicing')
        audio_args.add_argument('--subsampling', default=1, type=int)
        audio_args.add_argument('--sampling-rate', default=8000, type=int, help='sampling rate')
        audio_args.add_argument('--frame-size', default=1024, type=int, help='frame size')
        audio_args.add_argument('--frame-shift', default=256, type=int, help='frame shift')

        model_args = parser.add_argument_group('Model')
        model_args.add_argument('--model_file', required=True, help='model snapshot to use')
        model_args.add_argument('--num-speakers', type=int, default=4)
        model_args.add_argument('--hidden-size', default=256, type=int)
        model_args.add_argument('--transformer-encoder-n-heads', default=4, type=int)
        model_args.add_argument('--transformer-encoder-n-layers', default=2, type=int)

        vector_args = parser.add_argument_group('Model/EEND-vector-clustering')
        vector_args.add_argument('--spkv-dim', default=256, type=int)
        vector_args.add_argument(
            '--sil-spk-th', default=0.05, type=float,
            help='activity threshold to detect silent speaker'
        )
        vector_args.add_argument(
            '--ahc-dis-th', default=0.05, type=float,
            help='distance threshold above which clusters will not be merged'
        )
        vector_args.add_argument(
            '--clink-dis', default=1e+4, type=float,
            help='modified distance corresponding cannot-link'
        )
        vector_args.add_argument(
            '--num-clusters', default=-1, type=int,
            help='expecting number of clusters, any positive value will block --ahc-dis-th.',
        )
        decision_args = parser.add_argument_group('speaker active decision')
        decision_args.add_argument('--threshold', default=0.5, type=float)
        decision_args.add_argument('--median', default=1, type=int)

    @classmethod
    def from_args(cls, args) -> "DiarizationServeModel":
        """Initialize a DiarizationServeModel from argument Namespace."""
        eend_model = prepare_model_for_eval(args)
        return cls(
            eend_model,
            args.frame_size,
            args.frame_shift,
            args.input_transform,
            args.context_size,
            args.subsampling,
            args.chunk_size,
            sil_spk_th=args.sil_spk_th,
            ahc_dis_th=args.ahc_dis_th,
            clink_dis=args.clink_dis,
        )

    def _get_feature(self, audio_wav):
        """Transfrom audio wav into speech featrues."""
        Y = feature.stft(audio_wav, self.frame_size, self.frame_shift)
        Y = feature.transform(Y, transform_type=self.input_transform)
        Y = feature.splice(Y, context_size=self.context_size)
        Y = Y[::self.subsampling]
        return Y

    def _predict(self, audio_wav):
        """Return model prediction on audio wav."""
        acti_lst = []
        svec_lst = []

        Y = self._get_feature(audio_wav)

        with torch.no_grad():
            for start, end in _gen_chunk_indices(len(Y), self.chunk_size):
                if start > 0 and start + self.chunk_size > end:
                    # Ensure last chunk size
                    Y_chunked = torch.from_numpy(Y[end-self.chunk_size:end])
                else:
                    Y_chunked = torch.from_numpy(Y[start:end])
                Y_chunked = Y_chunked.to('cuda')

                outputs = self._model.batch_estimate(
                    torch.unsqueeze(Y_chunked, 0)
                )
                ys = outputs[0]

                for i in range(self.num_speakers):
                    spkivecs = outputs[i+1]
                    svec_lst.append(spkivecs[0].cpu().detach().numpy())

                if start > 0 and start + self.chunk_size > end:
                    # Ensure last chunk size
                    ys = list(ys)
                    ys[0] = ys[0][self.chunk_size-(end-start):self.chunk_size]

                acti = ys[0].cpu().detach().numpy()
                acti_lst.append(acti)

        acti_arr = np.array(acti_lst)
        svec_arr = np.array(svec_lst)
        return acti_arr, svec_arr

    def _rttm_generator(
        self, session, T_hat, threshold, median, sampling_rate
    ) -> Iterable[str]:
        """Create rttm string list from sequential speaker tagging."""
        actives = np.where(T_hat > threshold, 1, 0)
        if median > 1:
            actives = medfilt(actives, (median, 1))
        rttm_fmt = "SPEAKER {:s} 1 {:7.2f} {:7.2f} <NA> <NA> {:s} <NA>"
        frame_duration = self.frame_shift * self.subsampling / sampling_rate
        for spk_id, frames in enumerate(actives.T):
            frames = np.pad(frames, (1, 1), 'constant')
            changes, = np.where(np.diff(frames, axis=0) != 0)
            for s, e in zip(changes[::2], changes[1::2]):
                yield rttm_fmt.format(
                    session,
                    s * frame_duration,
                    (e - s) * frame_duration,
                    session + "_" + str(spk_id)
                )

    def diarize_wav(
        self,
        audio_wav,
        sampling_rate,
        threshold,
        median,
        num_clusters=-1,
        session="audio",
    ) -> List[str]:
        """Diarize audio wav into rttm string list.

        This is basically a pipe of function infer and make_rttm.
        """
        st_time = time.time()
        acti, svec = self._predict(audio_wav)
        eend_time = time.time()
        n_chunks = len(acti)
        # initialize clustering setting
        if num_clusters > 0:
            cls_num = num_clusters
            ahc_dis_th = None
        else:
            cls_num = None
            ahc_dis_th = self.ahc_dis_th
        # Get cannot-link index list and silence index list
        cl_lst, sil_lst = get_cl_sil(
            acti, cls_num, self.num_speakers, self.sil_spk_th
        )

        n_samples = n_chunks * self.num_speakers - len(sil_lst)
        min_n_samples = 2
        if cls_num is not None and cls_num > min_n_samples:
            min_n_samples = cls_num

        if n_samples >= min_n_samples:
            # clustering (if cls_num is None, update cls_num)
            clslab, cls_num = clustering(
                svec, cls_num, ahc_dis_th, cl_lst, sil_lst,
                self.clink_dis, self.num_speakers
            )
            # merge
            acti, clslab = merge_acti_clslab(acti, clslab, cls_num)
            # stitching
            out_chunks = stitching(acti, clslab, cls_num, self.num_speakers)
        else:
            out_chunks = acti
        T_hat = np.vstack(out_chunks)
        # begin make_rttm
        pred_time = time.time()
        rttm_list = [
            rttm_str for rttm_str in self._rttm_generator(
                session, T_hat, threshold, median, sampling_rate
            )
        ]
        ed_time = time.time()
        timing = {
            "eend": eend_time - st_time,
            "clustering": pred_time - eend_time,
            "rttm": ed_time - pred_time,
        }
        return rttm_list, timing

    def diarize_audio(
        self,
        audio_path,
        *args,
        **kwargs,
    ) -> List[str]:
        """Diarize audio file into rttm string list.

        This is basically a pipe of function infer and make_rttm.
        """
        session, _ = os.path.splitext(os.path.basename(audio_path))
        print(f"Start audio diarizing session: {session}")
        audio_wav, sampling_rate = load_wav(audio_path)
        return self.diarize_wav(audio_wav, sampling_rate, *args, **kwargs, session=session)


def _get_parser(local=True):
    parser = yamlargparse.ArgumentParser(
        description="EEND-vector-clustering demo"
    )
    parser.add_argument(
        '-c', '--config', help='config file path',
        action=yamlargparse.ActionConfigFile
    )
    DiarizationServeModel.add_args(parser)
    if local:
        parser.add_argument(
            '--audio_file',
            required=True,
            help='audio file to perform diarization.'
        )
    return parser


def main():
    parser = _get_parser()
    args = parser.parse_args()
    st_time = time.time()
    diarization_model = DiarizationServeModel.from_args(args)
    print(f"diarization_model is ready to run inference!")
    load_time = time.time()
    rst, timing = diarization_model.diarize_audio(
        args.audio_file,
        args.threshold,
        args.median,
        num_clusters=args.num_clusters,
    )
    timing["load_model"] = load_time - st_time
    rst_str = json.dumps(rst)
    print(rst_str)
    print(f"timing: {timing}")


if __name__ == "__main__":
    main()