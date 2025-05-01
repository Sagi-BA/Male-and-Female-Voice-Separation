#sagi changes 01_05_2025
# https://dashboard.pyannote.ai/
# https://huggingface.co/pyannote/segmentation-3.0
import asyncio
import streamlit as st
import numpy as np
import io
import os
import base64
import tempfile
import time
from pathlib import Path
import torch
# import whisper  # Removed whisper import
import librosa
import soundfile as sf
from pydub import AudioSegment
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import traceback
from demucs.pretrained import get_model
from demucs.apply import apply_model
import torchaudio
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from pyannote.core import Segment
from utils.counter import increment_user_count, get_user_count
from utils.init import initialize

# Fix for asyncio error
def setup_asyncio():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

# Initialize asyncio
loop = setup_asyncio()

# Load environment variables from .env file
load_dotenv()

# Set page configuration
st.set_page_config(
    initial_sidebar_state="collapsed",
    page_title="VoiceSplit - הפרדת קולות",
    page_icon="🎤",
    layout="wide",
)

# Initialize session state if not exists
if 'state' not in st.session_state:
    st.session_state.state = {
        'counted': False        
    }

# CUDA setup and device information
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    # Enable TF32 for better performance
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Set deterministic algorithms for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Fix for torch path issue
def setup_torch():
    try:
        # Initialize torch with custom settings
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        # Disable torch's internal file watcher
        torch.utils.data._utils.worker._worker_loop = None
    except Exception as e:
        st.warning(f"Warning: Could not fully initialize torch: {str(e)}")

# Initialize torch
setup_torch()

from torch.amp import autocast

def get_gpu_info():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        cuda_capability = torch.cuda.get_device_capability(0)
        return {
            "name": gpu_name,
            "total_memory": f"{total_memory:.2f}GB",
            "memory_used": f"{memory_allocated:.2f}GB",
            "cuda_version": torch.version.cuda,
            "cuda_capability": f"{cuda_capability[0]}.{cuda_capability[1]}"
        }
    return None

if torch.cuda.is_available():
    gpu_info = get_gpu_info()
    print(f"Using GPU: {gpu_info['name']}")
    print(f"CUDA Version: {gpu_info['cuda_version']}")
    print(f"CUDA Capability: {gpu_info['cuda_capability']}")
    print(f"Total GPU Memory: {gpu_info['total_memory']}")
else:
    print("CUDA is not available. Using CPU.")

# Utility: Convert MP3 to WAV if needed
def convert_to_wav(input_path):
    if input_path.lower().endswith(".mp3"):
        audio = AudioSegment.from_mp3(input_path)
        wav_path = input_path.replace(".mp3", ".wav")
        audio.export(wav_path, format="wav")
        return wav_path
    return input_path

@st.cache_resource
def load_demucs_model():
    try:
        model = get_model('htdemucs')
        model.to(DEVICE)
        return model
    except Exception as e:
        st.error(f"Error loading Demucs model: {str(e)}")
        return None

@st.cache_resource
def load_diarization_pipeline():
    try:
        # Get token from environment or UI
        huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
        
        if not huggingface_token:
            huggingface_token = st.text_input(
                "Enter your Hugging Face token:",
                type="password",
                help="Get your token from huggingface.co/settings/tokens"
            )
            if not huggingface_token:
                st.error("Please enter your Hugging Face token")
                return None
        
        # Try to load the diarization model
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",  # Changed from 3.1 to 3.0
                use_auth_token=huggingface_token
            )
            pipeline.to(DEVICE)
            return pipeline
        except Exception as e:
            st.error(f"Error loading diarization model: {str(e)}")
            st.error("Please make sure you have accepted the terms at: https://huggingface.co/pyannote/speaker-diarization-3.0")
            
            # Try alternative model if the first one fails
            try:
                st.info("Trying alternative model...")
                pipeline = Pipeline.from_pretrained(
                    "pyannote/segmentation-3.0",
                    use_auth_token=huggingface_token
                )
                pipeline.to(DEVICE)
                return pipeline
            except Exception as e2:
                st.error(f"Error loading segmentation model: {str(e2)}")
                st.error("Please make sure you have accepted the terms at: https://huggingface.co/pyannote/segmentation-3.0")
                return None

    except Exception as e:
        st.error(f"Error initializing pipeline: {str(e)}")
        st.error("\nPlease make sure:")
        st.error("1. You have accepted the terms at https://huggingface.co/pyannote/speaker-diarization-3.0")
        st.error("2. Your HuggingFace token is valid")
        return None

def extract_segment(waveform, start_sec, end_sec, sr):
    start_frame = int(start_sec * sr)
    end_frame = int(end_sec * sr)
    return waveform[:, start_frame:end_frame]

def process_audio(temp_path):
    try:
        # Load audio file and save a copy as original
        audio, sr = torchaudio.load(temp_path)
        original_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        torchaudio.save(original_file.name, audio, sr)
        
        # Convert to stereo if mono
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)  # Duplicate mono channel to create stereo
        elif audio.shape[0] > 2:
            audio = audio[:2]  # Keep only first two channels if more than stereo
        
        # Reshape audio to match Demucs requirements: (batch, channels, length)
        audio = audio.unsqueeze(0)  # Add batch dimension
        
        # Ensure sample rate matches model's expected rate (44.1kHz for Demucs)
        if sr != 44100:
            resampler = torchaudio.transforms.Resample(sr, 44100)
            audio = resampler(audio)

        # Move to appropriate device
        audio = audio.to(DEVICE)

        # First step: Extract vocals using Demucs
        model = load_demucs_model()
        if not model:
            return None

        with torch.no_grad():
            sources = apply_model(model, audio, device=DEVICE)
        
        # Get vocals and convert to mono
        vocals = sources[0][3]  # Index 3 corresponds to vocals
        vocals_mono = torch.mean(vocals, dim=0, keepdim=True)
        
        # Save vocals temporarily for diarization
        vocals_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        torchaudio.save(vocals_temp.name, vocals_mono.cpu(), 44100)

        # Second step: Perform speaker diarization
        diarization = load_diarization_pipeline()
        if not diarization:
            return None

        # Get diarization results
        diarization_results = diarization(vocals_temp.name)
        
        # Create separate audio for each speaker
        speakers_segments = {}
        for turn, _, speaker in diarization_results.itertracks(yield_label=True):
            if speaker not in speakers_segments:
                speakers_segments[speaker] = []
            speakers_segments[speaker].append((turn.start, turn.end))

        # Sort speakers by total duration to identify main speakers
        speaker_durations = {
            speaker: sum(end - start for start, end in segments)
            for speaker, segments in speakers_segments.items()
        }
        main_speakers = sorted(speaker_durations.items(), key=lambda x: x[1], reverse=True)[:2]

        # Extract and combine segments for each main speaker
        speaker_files = {}
        timeline_data = []
        
        # Save original audio
        speaker_files['original'] = original_file.name
        timeline_data.append({
            "speaker": "Original Audio",
            "start": 0,
            "end": 30
        })

        # Get total duration of the audio
        total_duration = vocals_mono.shape[1] / 44100  # Convert samples to seconds

        # Process main speakers (assuming first is male, second is female based on duration)
        speaker_labels = ["male_voice", "female_voice"]
        for (speaker, _), label in zip(main_speakers, speaker_labels):
            segments = speakers_segments[speaker]
            
            # Create a tensor of zeros with the same length as the original audio
            speaker_audio = torch.zeros_like(vocals_mono)
            
            # Fill in the segments where this speaker is talking
            for start, end in segments:
                start_sample = int(start * 44100)
                end_sample = int(end * 44100)
                segment_audio = vocals_mono[:, start_sample:end_sample]
                speaker_audio[:, start_sample:end_sample] = segment_audio

            # Save the speaker's audio with silence in non-speaking segments
            speaker_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            torchaudio.save(speaker_file.name, speaker_audio.cpu(), 44100)
            speaker_files[label] = speaker_file.name
            timeline_data.append({
                "speaker": label.replace('_', ' ').title(),
                "start": 0,
                "end": total_duration
            })

        # Clean up temporary files
        try:
            os.unlink(vocals_temp.name)
        except:
            pass

        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "speaker_files": speaker_files,
            "timeline_data": timeline_data
        }

    except Exception as e:
        st.error(f"Error processing file:\n{traceback.format_exc()}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None

# Custom CSS for better UI
st.markdown("""
<style>
    body {
        direction: rtl;
        text-align: right;
    }
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        height: 3rem;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .status-box {
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
    }
    .file-upload {
        border: 2px dashed #ccc;
        padding: 2rem;
        text-align: center;
        border-radius: 4px;
        background-color: #f8f9fa;
    }
    .results-section {
        margin-top: 2rem;
        padding: 1rem;
        border-radius: 4px;
        background-color: #fff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .timeline-chart {
        margin: 2rem 0;
        padding: 1rem;
        border-radius: 4px;
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if 'speaker_audios' not in st.session_state:
        st.session_state.speaker_audios = {}
    if 'timeline_data' not in st.session_state:
        st.session_state.timeline_data = []
    if 'progress' not in st.session_state:
        st.session_state.progress = 0

def create_download_link(data, filename, text):
    """Create a download link for a file"""
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}" class="download-button">{text}</a>'
    return href

def create_timeline_chart(timeline_data):
    if not timeline_data:
        st.warning("🔍 לא נמצאו דוברים – לא ניתן להציג גרף.")
        return go.Figure()

    df = pd.DataFrame(timeline_data)

    # Translate speaker names to Hebrew
    df['speaker'] = df['speaker'].replace({
        'Original Audio': 'שמע מקורי',
        'Male Voice': 'קול גבר',
        'Female Voice': 'קול אישה'
    })

    speakers = df['speaker'].unique()
    colors = px.colors.qualitative.Set3[:len(speakers)]
    color_map = dict(zip(speakers, colors))

    fig = go.Figure()
    for speaker in speakers:
        speaker_data = df[df['speaker'] == speaker]
        fig.add_trace(go.Bar(
            name=speaker,
            x=speaker_data['end'] - speaker_data['start'],
            y=[speaker] * len(speaker_data),
            orientation='h',
            base=speaker_data['start'],
            marker_color=color_map[speaker],
            customdata=speaker_data[['start', 'end']],
            hovertemplate='%{customdata[0]:.1f}s - %{customdata[1]:.1f}s<br>משך: %{x:.1f}s<extra></extra>'
        ))

    fig.update_layout(
        title="🔊 ציר זמן לפי דוברים",
        xaxis_title="זמן (שניות)",
        yaxis_title="דובר",
        barmode='overlay',
        height=300,
        plot_bgcolor='white',
        paper_bgcolor='white',
        # RTL support
        xaxis=dict(
            side='top',
            autorange='reversed'
        ),
        yaxis=dict(
            side='right'
        )
    )

    return fig

def file_uploader_with_path():
    """Custom file uploader that handles file paths correctly"""
    uploaded_file = st.file_uploader(
        "בחר קובץ שמע", label_visibility="collapsed",
        type=["mp3", "wav", "m4a"],  # Changed from set to list
        accept_multiple_files=False,
        key="audio_uploader"
    )
    
    if uploaded_file is not None:
        try:
            # Get the file extension from the uploaded file
            file_ext = uploaded_file.name.split('.')[-1].lower()
            
            # Create a temporary file with the correct extension
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}', mode='wb') as tmp_file:
                # Read the uploaded file in chunks to handle large files
                CHUNK_SIZE = 1024 * 1024  # 1MB chunks
                while True:
                    chunk = uploaded_file.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    tmp_file.write(chunk)
                
                # Ensure all data is written to disk
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
                
                return tmp_file.name, uploaded_file
        except Exception as e:
            st.error(f"Error processing uploaded file: {str(e)}")
            return None, None
    return None, None

def check_ffmpeg_installation():
    """Check if ffmpeg is installed and available in PATH"""
    try:
        import subprocess
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def convert_to_mp3(wav_path, bitrate='64k'):
    """Convert WAV file to MP3 with specified bitrate"""
    try:
        # Set custom ffmpeg path if available
        if os.path.exists('/usr/bin/ffmpeg'):
            AudioSegment.converter = '/usr/bin/ffmpeg'
        elif os.path.exists('/usr/local/bin/ffmpeg'):
            AudioSegment.converter = '/usr/local/bin/ffmpeg'
        
        # Load the WAV file
        audio = AudioSegment.from_wav(wav_path)
        
        # Create MP3 file path
        mp3_path = wav_path.replace('.wav', '.mp3')
        
        # Export as MP3 with specified bitrate
        audio.export(mp3_path, format='mp3', bitrate=bitrate)
        
        return mp3_path
    except Exception as e:
        st.error(f"שגיאה בהמרת הקובץ ל-MP3: {str(e)}")
        st.error("""
        אנא וודא ש-ffmpeg מותקן בשרת:
        1. התחבר לשרת דרך SSH
        2. הרץ: sudo apt-get update
        3. הרץ: sudo apt-get install ffmpeg
        4. הפעל מחדש את האפליקציה
        """)
        return None

def hide_streamlit_header_footer():
    hide_st_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 0rem;}
    </style>
    """
    st.markdown(hide_st_style, unsafe_allow_html=True)

def load_html_file(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        return f.read()
    
def main():
    with st.spinner('האפליקציה נטענת...'):
        footer_content = initialize()
        # # st.title("🎨 מחולל התמונות החכם")
        hide_streamlit_header_footer()               

    initialize_session_state()

    st.title("🎤 VoiceSplit - הפרדת קולות גבר/אישה")
    
    # Display device and GPU information
    gpu_info = get_gpu_info()
    if gpu_info:
        # st.write("### 🎮 מידע על החומרה")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success(f"🚀 כרטיס מסך: {gpu_info['name']}")
        with col2:
            st.info(f"⚡ CUDA: {gpu_info['cuda_version']} (יכולת {gpu_info['cuda_capability']}")
        with col3:
            st.info(f"💾 זיכרון כרטיס מסך: {gpu_info['total_memory']}")
    # else:
    #     st.warning("🔧 הרצה על מעבד - העיבוד יהיה איטי יותר. לקבלת ביצועים טובים יותר, אנא וודא ש-CUDA מותקן כראוי.")

     # Load and display the custom expander HTML
    expander_html = load_html_file('expander.html')
    st.markdown(expander_html, unsafe_allow_html=True)    

    # File upload section
    st.markdown("### 📂 שלב 1: העלאת קובץ שמע")
    st.markdown("העלה את קובץ השמע שלך (פורמט MP3, WAV או M4A)")
    
    temp_path = None
    try:
        temp_path, uploaded_file = file_uploader_with_path()
        
        if temp_path and uploaded_file:
            try:
                # Verify the audio file can be opened before processing
                with sf.SoundFile(temp_path) as audio_file:
                    # Display audio player if file is valid
                    st.audio(uploaded_file, format=f'audio/{uploaded_file.name.split(".")[-1].lower()}')
                
                # Process button
                st.markdown("### ⚙️ שלב 2: עיבוד השמע")

                 # Add bitrate selection
                bitrate = st.selectbox(
                    "בחר איכות קובץ MP3:",
                    options=['64k', '96k', '128k', '192k', '256k'],
                    index=0,
                    help="איכות נמוכה יותר = גודל קובץ קטן יותר"
                )
                
                if st.button("🎯 הפרדת הקולות"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    with st.spinner("🔄 מעבד את השמע שלך..."):
                        status_text.text("⚡ טוען מודל Demucs...")
                        progress_bar.progress(20)
                        
                        status_text.text("🎤 מפריד קולות...")
                        progress_bar.progress(40)
                        
                        result = process_audio(temp_path)
                        
                        if result:
                            status_text.text("✅ עיבוד הושלם בהצלחה!")
                            progress_bar.progress(100)
                            
                            st.session_state.processed = True
                            st.session_state.speaker_audios = result["speaker_files"]
                            st.session_state.timeline_data = result["timeline_data"]
                            st.rerun()
                        else:
                            status_text.text("❌ שגיאה בעיבוד השמע")
                            progress_bar.progress(0)
            except sf.SoundFileError as e:
                st.error(f"שגיאה: קובץ שמע לא תקין או פגום. אנא נסה להעלות קובץ אחר. פרטים: {str(e)}")
            except Exception as e:
                st.error(f"שגיאה בעיבוד קובץ השמע: {str(e)}")
    finally:
        # Clean up temporary file when done
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                st.warning(f"אזהרה: לא ניתן למחוק את הקובץ הזמני: {str(e)}")

    # Results section
    if st.session_state.processed:
        st.markdown("### 📊 שלב 3: תוצאות")
        
        # Display timeline visualization
        st.markdown("#### 📈 ציר זמן של השמע")
        timeline_chart = create_timeline_chart(st.session_state.timeline_data)
        st.plotly_chart(timeline_chart, use_container_width=True)

        # Display and download audio files
        st.markdown("#### 🔊 קבצי שמע")
        for name, audio_path in st.session_state.speaker_audios.items():
            try:
                if os.path.exists(audio_path):
                    display_name = name.replace('_', ' ').title()
                    if display_name == "Original Audio":
                        display_name = "שמע מקורי"
                    elif display_name == "Male Voice":
                        display_name = "קול גבר"
                    elif display_name == "Female Voice":
                        display_name = "קול אישה"
                        
                    st.markdown(f"**{display_name}**")
                    
                    # Convert to MP3
                    mp3_path = convert_to_mp3(audio_path, bitrate)
                    if mp3_path:
                        with open(mp3_path, 'rb') as audio_file:
                            audio_bytes = audio_file.read()
                            st.audio(audio_bytes, format='audio/mp3')
                            st.markdown(
                                create_download_link(audio_bytes, f"{name}.mp3", f"⬇️ הורד {display_name} (MP3)"),
                                unsafe_allow_html=True
                            )
                        # Clean up MP3 file
                        try:
                            os.unlink(mp3_path)
                        except:
                            pass
            finally:
                # Clean up temporary audio files
                if os.path.exists(audio_path):
                    try:
                        os.unlink(audio_path)
                    except Exception as e:
                        st.warning(f"אזהרה: לא ניתן למחוק את קובץ השמע הזמני: {str(e)}")

    # Display footer content
    st.markdown(footer_content, unsafe_allow_html=True)    

    # Display user count
    user_count = get_user_count(formatted=True)
    # print(user_count)
    st.markdown(f"<p class='user-count' style='color: #4B0082;'>סה\"כ משתמשים: {user_count}</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    # Increment user count on first load
    if 'counted' not in st.session_state:
        st.session_state.counted = True
        increment_user_count()

    main()