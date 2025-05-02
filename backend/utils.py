def download_from_drive(file_id, destination):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, destination, quiet=False)

def download_from_gdrive(file_id, destination):
    print(f"Downloading model from Google Drive: {file_id}")
    session = requests.Session()
    URL = "https://drive.google.com/uc?export=download"

    response = session.get(URL, params={"id": file_id}, stream=True)
    confirm_token = _get_confirm_token(response)

    if confirm_token:
        response = session.get(URL, params={"id": file_id, "confirm": confirm_token}, stream=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None

def create_fbank():
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.remove_dc_offset = False
    opts.frame_opts.window_type = "hann"

    opts.mel_opts.low_freq = 0
    opts.mel_opts.num_bins = 80

    opts.mel_opts.is_librosa = True

    fbank = knf.OnlineFbank(opts)
    return fbank


def compute_features(audio, fbank):
    assert len(audio.shape) == 1, audio.shape
    fbank.accept_waveform(16000, audio)
    ans = []
    processed = 0
    while processed < fbank.num_frames_ready:
        ans.append(np.array(fbank.get_frame(processed)))
        processed += 1
    ans = np.stack(ans)
    return ans
