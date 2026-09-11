"""Unadopted, task-specific observation-prefix I/O candidate; activation is explicit."""
from contextlib import contextmanager
import functools


@contextmanager
def observation_prefix_reads():
    from robomimic.utils.dataset import SequenceDataset
    from robomimic.utils.ot_dataset import ChunkSamplingDTWTrainDataset
    originals = {cls: cls.get_obs_sequence_from_demo
                 for cls in [SequenceDataset, ChunkSamplingDTWTrainDataset]}
    for cls, original in originals.items():
        @functools.wraps(original)
        def shorter(self, *args, _original=original, **kwargs):
            assert kwargs["prefix"] == "obs" and kwargs["num_frames_to_stack"] == 1
            assert kwargs["seq_length"] == 16
            # History is one prepended frame plus the current frame. The model
            # consumes precisely this prefix; action sequences remain length17.
            kwargs["seq_length"] = 1
            return _original(self, *args, **kwargs)
        cls.get_obs_sequence_from_demo = shorter
    try:
        yield
    finally:
        for cls, original in originals.items():
            cls.get_obs_sequence_from_demo = original
