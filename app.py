from pathlib import Path

import librosa
import streamlit as st
import torch
from streamlit.runtime.uploaded_file_manager import UploadedFile
from torch import Tensor

from src.model import VoiceClassifyModel
from src.spec_process import SpectrogramProcessor
from src.train import device_detected


def load_saved_model(model_path: Path) -> tuple[VoiceClassifyModel, torch.device]:
    """ Load the model state dict from **model_path**
    :param model_path: tht model state path
    :return: the model that pretrained and user device
    """
    device = torch.device(device_detected())
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model = VoiceClassifyModel(classify_class=2)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device

def read_all_model():
    ...

def predict(model: VoiceClassifyModel, device: torch.device, voice_file: UploadedFile) -> tuple[Tensor, float]:
    """Predict the voice file by pretrained model.
    :param model: model have pretrained argument.
    :param device: user device
    :param voice_file: the voice file made by streamlit input.
    :return: the probability tensor and pred index (0: male, 1: female)
    """
    with torch.no_grad():
        with st.spinner('Wait for process', show_time=True):
            process = SpectrogramProcessor()
            y, sr = librosa.load(voice_file, sr=process.sr)
            y, _ = librosa.effects.trim(y)

            x = process.transform_to_tensor(y).to(device)
            x = x.unsqueeze(0)
            out = model(x)
            prob = torch.nn.functional.softmax(out, dim=1)
            pred = torch.argmax(prob, dim=1).item()
    return prob, pred


def app_entry(model: VoiceClassifyModel, device: torch.device):
    st.title("Voice Classification")

    input_c1, input_c2 = st.columns([1, 1])
    upload_file = record_file = None

    with input_c1:
        record_file = st.audio_input("Record a voice message")
        if record_file:
            st.audio(record_file)
            st.success('Record success!')

    with input_c2:
        upload_file = st.file_uploader(label='Voice file', type=['mp3', 'wav', 'aac'])
        if upload_file:
            st.success('Upload success!')

    select_source = st.radio(
        'Select the analyze file: ',
        ['Record', 'Upload']
    )

    voice_file = record_file if select_source == 'Record' else upload_file

    if st.button('Analyze', type='primary'):
        if not voice_file:
            st.error('Please record a voice or upload the voice file.')
            return

        prob, pred = predict(model, device, voice_file)

        # Predict Result
        st.subheader('Predict Result')
        c1, c2 = st.columns([1.1, 2])
        with c1:
            st.write(f'The gender that model predicted is')
        with c2:
            gender = 'male' if pred == 0 else 'female'
            gender_color = 'green' if gender == 'male' else 'red'
            st.badge(gender, color=gender_color)

        # Model confidence
        c1, c2 = st.columns([1.1, 2])
        with c1:
            st.write(f'Model confidence: ')
        with c2:
            confidence = prob[0, pred].item() * 100
            if confidence > 66:
                icon, color = ':material/check:', 'green'
            elif confidence > 33:
                icon, color = ':material/do_not_disturb_on:', 'yellow'
            else:
                icon, color = ':material/close:', 'red'
            st.badge(f'{confidence:.2f}%', icon=icon, color=color)

    st.caption('powered by Andongni0723')


def main():
    model, device = load_saved_model(Path('model/model_state_data_count_5000.pth'))
    app_entry(model, device)


if __name__ == '__main__':
    main()
