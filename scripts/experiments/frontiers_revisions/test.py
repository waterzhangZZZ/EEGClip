from EEGClip.clip_models import EEGClipModel
import torch
eegclipmodel = EEGClipModel()
EEGEncoder = torch.nn.Sequential(
    eegclipmodel.eeg_encoder, eegclipmodel.eeg_projection
)

print(EEGEncoder)