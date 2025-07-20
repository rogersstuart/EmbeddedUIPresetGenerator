"""
Parameter testing application with audio recording and MIDI playback.
Automatically generates synthesizer presets by testing parameter combinations.
"""

import time
import mido
import sounddevice as sd
import soundfile as sf
import numpy as np
import json
import csv
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Set, List, Any, Optional, Tuple
from dataclasses import dataclass
import serial
import argparse
import random


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='parameter_testing.log'
)
logger = logging.getLogger(__name__)


@dataclass
class TestConfig:
    """Configuration settings for parameter testing"""
    csv_filename: str = "restricted_parameter_data.csv"
    midi_port_index: int = 3
    sample_rate: int = 48000
    channels: int = 2
    device_id: int = 2
    serial_port: str = 'COM3'
    baudrate: int = 500000
    audio_threshold: float = 0.01
    sample_delay: float = 0.5
    
    @classmethod
    def from_args(cls, args):
        """Create TestConfig from command line arguments"""
        return cls(
            csv_filename=args.csv_file,
            midi_port_index=args.midi_port,
            sample_rate=args.sample_rate,
            channels=args.channels,
            device_id=args.audio_device,
            serial_port=args.com_port,
            baudrate=args.baudrate,
            audio_threshold=args.audio_threshold,
            sample_delay=args.sample_delay
        )


class DeviceManager:
    """Handles device listing and information"""
    
    @staticmethod
    def list_midi_devices() -> None:
        """List all available MIDI output devices with their indexes"""
        print("\nAvailable MIDI output devices:")
        print("-" * 40)
        devices = mido.get_output_names()
        for idx, device in enumerate(devices):
            print(f"{idx}: {device}")
        print()

    @staticmethod
    def list_audio_devices() -> None:
        """List all available audio input devices with their indexes"""
        print("\nAvailable audio input devices:")
        print("-" * 40)
        devices = sd.query_devices()
        for idx, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"{idx}: {device['name']} (Channels: {device['max_input_channels']})")
        print()


class AudioRecorder:
    """Handles audio recording functionality"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.audio_buffer: List[np.ndarray] = []
    
    def _audio_callback(self, indata: np.ndarray, frames: int, 
                       time_info: dict, status: sd.CallbackFlags) -> None:
        """Callback for audio recording"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        self.audio_buffer.append(indata.copy())
    
    def record_with_midi_notes(self, filename: str, midi_port_name: str) -> bool:
        """Record audio while playing MIDI notes"""
        try:
            self.audio_buffer = []
            
            with mido.open_output(midi_port_name) as outport:
                with sd.InputStream(samplerate=self.config.sample_rate,
                                  channels=self.config.channels,
                                  callback=self._audio_callback,
                                  device=self.config.device_id):
                    
                    logger.info(f"Recording audio to {filename}")
                    time.sleep(0.1)

                    # Play a note
                    outport.send(mido.Message('note_on', note=60, velocity=127))
                    time.sleep(10)
                    outport.send(mido.Message('note_off', note=60, velocity=0))
                    time.sleep(1)

                # Save audio
                audio_np = np.concatenate(self.audio_buffer, axis=0)
                sf.write(filename, audio_np, self.config.sample_rate)
                return True

        except Exception as e:
            logger.error(f"Recording failed: {e}")
            return False
    
    def record_with_midi_file(self, midi_filename: str, audio_filename: str, 
                            duration: float, midi_port_name: str) -> None:
        """Play a MIDI file and record audio for a fixed duration"""
        self.audio_buffer = []
        
        with mido.open_output(midi_port_name) as outport:
            mid = mido.MidiFile(midi_filename)

            with sd.InputStream(samplerate=self.config.sample_rate,
                              channels=self.config.channels,
                              callback=self._audio_callback,
                              device=self.config.device_id):
                
                logger.info(f"Recording {duration}s while playing {midi_filename}")
                start_time = time.time()
                
                # Play MIDI file
                for msg in mid.play():
                    outport.send(msg)
                    if time.time() - start_time > duration:
                        break
                
                # Wait for remaining duration if needed
                elapsed = time.time() - start_time
                if elapsed < duration:
                    time.sleep(duration - elapsed)

            # Save audio
            audio_np = np.concatenate(self.audio_buffer, axis=0)
            sf.write(audio_filename, audio_np, self.config.sample_rate)
            logger.info(f"Audio saved to {audio_filename}")
    
    @staticmethod
    def is_audio_silent(filename: str, threshold: float = 0.01) -> bool:
        """Check if an audio file contains only silence"""
        try:
            data, _ = sf.read(filename)
            rms = np.sqrt(np.mean(np.square(data)))
            logger.debug(f"Audio RMS level: {rms}")
            return rms < threshold
        except Exception as e:
            logger.error(f"Error checking audio: {e}")
            return False


class ParameterController:
    """Handles parameter communication with the synthesizer"""
    
    def __init__(self, serial_port: serial.Serial):
        self.ser = serial_port
    
    def set_param_value(self, param_num: int, value: int) -> None:
        """
        Sets parameter value using serial communication.
        For params 0-254: sends 's {param} {value}'
        For params 255-511: sends 's 255 {param-255} {value}'
        """
        cmd = 's'
        
        # Clear any pending data
        self.ser.reset_input_buffer()
        
        # Send command
        encoded = cmd.encode('ascii', errors='ignore')
        if param_num <= 254:
            encoded = encoded + bytes([param_num, value])
        else:
            encoded = encoded + bytes([255, param_num - 255, value])

        self.ser.write(encoded)
        self.ser.flush()
    
    def reset_to_default(self) -> None:
        """Reset synthesizer to default state"""
        self.ser.write(b'r 0')
        time.sleep(1)


class DataManager:
    """Handles CSV data operations and parameter specifications"""
    
    @staticmethod
    def read_param_specs_from_csv(csv_filename: str) -> Dict[int, List[int]]:
        """
        Read parameter specifications from a CSV file.
        
        Expected CSV format:
        param_num,value_spec
        0,"0,85,170,255"
        5,"0,127"
        10,"1,4,6,8"
        """
        param_specs = {}
        
        try:
            with open(csv_filename, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)  # Skip header row
                
                for row in reader:
                    if len(row) >= 2:
                        try:
                            param_num = int(row[0])
                            value_spec = row[1]
                            # Parse comma-separated values
                            values = [int(v.strip()) for v in value_spec.split(',')]
                            param_specs[param_num] = values
                        except ValueError:
                            logger.warning(f"Could not parse row {row}")
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            raise
        
        return param_specs
    
    @staticmethod
    def load_existing_param_jsons(csv_filename: str) -> Set[str]:
        """Load all parameter JSON strings from the CSV file into a set"""
        tried = set()
        if not os.path.exists(csv_filename):
            return tried
            
        with open(csv_filename, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                params = row.get('parameters')
                if params:
                    tried.add(params)
        return tried
    
    @staticmethod
    def get_last_sample_index(csv_filename: str) -> int:
        """Return the last used sample index from the CSV file, or -1 if not found"""
        if not os.path.exists(csv_filename):
            return -1
            
        last_index = -1
        with open(csv_filename, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                try:
                    idx = int(row['index'])
                    if idx > last_index:
                        last_index = idx
                except Exception:
                    continue
        return last_index
    
    @staticmethod
    def save_test_result(csv_filename: str, sample_index: int, 
                        param_dict: Dict[str, Any]) -> None:
        """Save test results to CSV file"""
        params_json = json.dumps(param_dict)
        
        # Initialize CSV with headers if it doesn't exist
        if not os.path.exists(csv_filename):
            with open(csv_filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['index', 'timestamp', 'parameters'])
                writer.writeheader()
        
        # Append data
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['index', 'timestamp', 'parameters'])
            writer.writerow({
                'index': sample_index,
                'timestamp': datetime.now().isoformat(),
                'parameters': params_json
            })


class ParameterTester:
    """Main class for parameter testing"""
    
    def __init__(self, config: TestConfig, param_specs: Dict[int, List[int]]):
        self.config = config
        self.param_specs = param_specs
        self.audio_recorder = AudioRecorder(config)
        self.data_manager = DataManager()
    
    def process_single_test(self, controller: ParameterController, 
                          tried_params: Set[str], sample_index: int) -> bool:
        """Process a single test iteration"""
        selected_values = {}
        
        try:
            # Select random values for each parameter
            for param_num, allowed_values in self.param_specs.items():
                value = random.choice(allowed_values)
                selected_values[param_num] = value
                
                logger.info(f"Setting parameter {param_num} to {value}")
                controller.set_param_value(param_num, value)
            
            # Check if combination already tested
            param_dict = {str(k): v for k, v in selected_values.items()}
            params_json = json.dumps(param_dict, sort_keys=True)
            
            if params_json in tried_params:
                logger.info("Combination already tested, skipping.")
                return False
            
            tried_params.add(params_json)
            
            # Record audio with simple notes
            wav_filename = f"{sample_index}.wav"
            midi_port_name = mido.get_output_names()[self.config.midi_port_index]
            
            if not self.audio_recorder.record_with_midi_notes(wav_filename, midi_port_name):
                logger.error(f"Failed to record audio for sample {sample_index}")
                return False
            
            # Reset to default
            controller.reset_to_default()
            
            # Check if audio is silent
            if self.audio_recorder.is_audio_silent(wav_filename, self.config.audio_threshold):
                logger.warning("Audio is silent, deleting file and skipping")
                try:
                    os.remove(wav_filename)
                except Exception as e:
                    logger.error(f"Error deleting file: {e}")
                return False
            
            # Record with MIDI file
            test_filename = f"{sample_index}_test.wav"
            self.audio_recorder.record_with_midi_file(
                "./test_synth.mid", test_filename, 10, midi_port_name
            )
            
            # Reset again
            controller.reset_to_default()
            
            # Save results
            self.data_manager.save_test_result(
                self.config.csv_filename, sample_index, param_dict
            )
            
            logger.info(f"Test {sample_index} completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error processing sample {sample_index}: {e}")
            return False
    
    def run_tests(self, controller: ParameterController, duration_hours: float) -> None:
        """Run parameter tests for specified duration"""
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        logger.info(f"Starting parameter testing until {end_time}")
        
        tried_params = self.data_manager.load_existing_param_jsons(self.config.csv_filename)
        sample_index = self.data_manager.get_last_sample_index(self.config.csv_filename) + 1
        
        try:
            while datetime.now() < end_time:
                if self.process_single_test(controller, tried_params, sample_index):
                    sample_index += 1
                    time.sleep(self.config.sample_delay)
                    
        except KeyboardInterrupt:
            logger.info("Testing interrupted by user")
        except Exception as e:
            logger.error(f"Testing failed: {e}")
            raise
        finally:
            logger.info(f"Testing completed. Processed {sample_index} samples")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Parameter testing application with audio recording and MIDI playback',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Device listing options
    parser.add_argument('--list-midi', action='store_true',
                        help='List all available MIDI output devices and exit')
    parser.add_argument('--list-audio', action='store_true',
                        help='List all available audio input devices and exit')
    parser.add_argument('--list-all', action='store_true',
                        help='List all available devices (audio and MIDI) and exit')
    
    # Device configuration
    parser.add_argument('--midi-port', type=int, default=3,
                        help='MIDI output port index')
    parser.add_argument('--audio-device', type=int, default=2,
                        help='Audio input device index')
    parser.add_argument('--com-port', type=str, default='COM3',
                        help='Serial COM port')
    parser.add_argument('--baudrate', type=int, default=500000,
                        help='Serial port baudrate')
    
    # Audio configuration
    parser.add_argument('--sample-rate', type=int, default=48000,
                        help='Audio sample rate in Hz')
    parser.add_argument('--channels', type=int, default=2,
                        help='Number of audio channels')
    parser.add_argument('--audio-threshold', type=float, default=0.01,
                        help='RMS threshold for silence detection')
    
    # Test configuration
    parser.add_argument('--duration', type=float, default=24,
                        help='Test duration in hours')
    parser.add_argument('--sample-delay', type=float, default=0.5,
                        help='Delay between samples in seconds')
    parser.add_argument('--csv-file', type=str, default='restricted_parameter_data.csv',
                        help='Output CSV filename')
    parser.add_argument('--param-specs', type=str, default='param_specs.csv',
                        help='Parameter specifications CSV filename')
    
    # Logging
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    
    return parser.parse_args()


def setup_logging(debug: bool = False) -> None:
    """Configure logging settings"""
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)


def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Handle device listing
    device_manager = DeviceManager()
    if args.list_all or args.list_midi:
        device_manager.list_midi_devices()
    if args.list_all or args.list_audio:
        device_manager.list_audio_devices()
    if args.list_all or args.list_midi or args.list_audio:
        return
    
    # Set up logging
    setup_logging(debug=args.debug)
    
    # Load parameter specifications
    data_manager = DataManager()
    param_specs = data_manager.read_param_specs_from_csv(args.param_specs)
    logger.info(f"Loaded {len(param_specs)} parameters from {args.param_specs}")
    
    # Create configuration
    config = TestConfig.from_args(args)
    
    # Log configuration
    logger.info("Configuration:")
    logger.info(f"  COM Port: {config.serial_port}")
    logger.info(f"  MIDI Port Index: {config.midi_port_index}")
    logger.info(f"  Audio Device Index: {config.device_id}")
    logger.info(f"  Sample Rate: {config.sample_rate}")
    logger.info(f"  Output CSV: {config.csv_filename}")
    
    # Run tests
    try:
        with serial.Serial(config.serial_port, baudrate=config.baudrate, timeout=1) as ser:
            time.sleep(0.5)  # Wait for port to initialize
            ser.reset_input_buffer()
            
            controller = ParameterController(ser)
            tester = ParameterTester(config, param_specs)
            tester.run_tests(controller, args.duration)
            
    except serial.SerialException as e:
        logger.error(f"Serial port error: {e}")
        raise
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise


if __name__ == "__main__":
    main()