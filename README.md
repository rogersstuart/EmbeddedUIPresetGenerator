# EmbeddedUI Preset Generator

A comprehensive tool for automatically generating and classifying synthesizer presets using parameter testing, audio recording, and AI-powered audio classification.

## Overview

This project consists of two main components:

1. **PresetGenerator.py** - Automatically generates synthesizer presets by testing parameter combinations, recording audio output, and storing results
2. **AudioClassification.py** - Analyzes recorded audio files using CLAP (Contrastive Language-Audio Pre-training) and finds best matches based on audio characteristics

## Features

### PresetGenerator.py
- Automated parameter testing with configurable value ranges
- Serial communication with synthesizer hardware
- MIDI playback and audio recording
- Silence detection to skip invalid parameter combinations
- Resume capability from previous test sessions
- Comprehensive logging and error handling

### AudioClassification.py
- Audio classification using CLAP model from Hugging Face
- Multi-category audio analysis across 17 different characteristics
- LLM-powered matching with similarity expansion
- CSV-based data management
- GPU acceleration support

## Requirements

### Hardware
- XFM/XVA Synthesizer
- MIDI output device
- Audio input device

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure your hardware settings in the scripts or via command line arguments

## Usage

### PresetGenerator.py

Generate synthesizer presets by testing parameter combinations:

```bash
# List available devices
python PresetGenerator.py --list-all

# Run preset generation
python PresetGenerator.py --duration 24 --com-port COM3 --midi-port 3 --audio-device 2

# With custom parameter specifications
python PresetGenerator.py --param-specs my_params.csv --output presets.csv
```

#### Command Line Arguments:
- `--list-midi`: List MIDI devices
- `--list-audio`: List audio devices  
- `--list-all`: List all devices
- `--midi-port`: MIDI output port index (default: 3)
- `--audio-device`: Audio input device index (default: 2)
- `--com-port`: Serial COM port (default: COM3)
- `--baudrate`: Serial baudrate (default: 500000)
- `--sample-rate`: Audio sample rate (default: 48000)
- `--channels`: Audio channels (default: 2)
- `--audio-threshold`: Silence detection threshold (default: 0.01)
- `--duration`: Test duration in hours (default: 24)
- `--sample-delay`: Delay between samples (default: 0.5s)
- `--csv-file`: Output CSV filename (default: restricted_parameter_data.csv)
- `--param-specs`: Parameter specifications CSV (default: param_specs.csv)

#### Parameter Specifications CSV Format:
```csv
param_num,value_spec
0,"0,85,170,255"
5,"0,127"
10,"1,4,6,8"
```

### AudioClassification.py

Classify recorded audio files and find matches:

```bash
# Run audio classification on all WAV files
python AudioClassification.py --classify

# Find best match for a specific prompt
python AudioClassification.py --match "ambient strings"

# Specify custom directories
python AudioClassification.py --classify --audio-dir ./my_audio --output-csv ./results.csv
```

#### Command Line Arguments:
- `--classify`: Run audio classification on WAV files
- `--match`: Find best match for given prompt
- `--audio-dir`: Directory containing audio files
- `--output-csv`: Output CSV file path

## Audio Classification Categories

The AudioClassification script analyzes audio across 17 categories:

1. **Type** - Sound type (stab, riser, pluck)
2. **Noise** - Noise characteristics
3. **Instrument** - Instrument similarity
4. **Synth** - Synthesizer characteristics
5. **Tone Quality** - Tonal qualities (bright, dark, warm, etc.)
6. **Pitch Level** - Pitch characteristics
7. **Volume Intensity** - Volume and intensity descriptors
8. **Time Shape** - Temporal characteristics
9. **Sound Texture** - Textural qualities
10. **Emotional Feel** - Emotional associations
11. **More Emotion** - Additional emotional descriptors
12. **Sound Source** - Sound generation source
13. **Genre Style** - Musical genre associations
14. **Spatial Sense** - Spatial characteristics
15. **Motion Character** - Movement qualities
16. **Clarity Quality** - Clarity and quality descriptors
17. **Rhythmic Flow** - Rhythmic characteristics

## Workflow

1. **Parameter Testing**:
   - Define parameter ranges in CSV
   - Run PresetGenerator.py to test combinations
   - Audio files are recorded for each valid combination

2. **Audio Classification**:
   - Run AudioClassification.py with `--classify` flag
   - Each audio file is analyzed across all categories
   - Results are saved to CSV

3. **Preset Matching**:
   - Use `--match` with descriptive prompt
   - LLM expands search terms for better matching
   - Returns best matching preset index

## Output Files

- **Audio Files**: `{index}.wav` and `{index}_test.wav` for each preset
- **Parameter Data**: CSV with index, timestamp, and parameter values
- **Classification Data**: CSV with index, parameters, and classification string

## Notes

- The serial communication protocol expects specific byte formats for parameter control
- Audio silence detection prevents saving invalid presets
- GPU acceleration is automatically used when available
- LLM integration requires a local LM Studio instance or OpenAI API key