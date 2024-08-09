import torch


def mix_stems(stems):
    return sum(stems)


def mix_down(waveform):
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    return waveform


def right_pad(waveform, expected_num_frames):
    num_frames = waveform.shape[1]
    if num_frames < expected_num_frames:
        num_missing_frames = expected_num_frames - num_frames
        last_dim_padding = (0, num_missing_frames)
        waveform = torch.nn.functional.pad(waveform, last_dim_padding)
    return waveform
