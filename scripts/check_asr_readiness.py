"""Check real optional packages/PCM decoding; never load Whisper weights."""
import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from traveler_agent.academic_acceptance import write_report
from traveler_agent.audio_asr import main as diagnose, validate_wav


def readiness():
    buffer = io.StringIO()
    with redirect_stdout(buffer): diagnose(['--check'])
    report = json.loads(buffer.getvalue())
    report.update(scope='REAL_PACKAGES_AND_PCM_DECODER_NOT_REAL_SPEECH_RECOGNITION',
                  whisper_weights='NOT_LOADED_OR_DOWNLOADED', windows_microphone='PENDING_USER_TEST')
    try:
        import numpy as np
        from faster_whisper.audio import decode_audio
        with tempfile.TemporaryDirectory(prefix='traveler-asr-readiness-') as temporary:
            path = Path(temporary) / 'generated-non-speech.wav'
            with wave.open(str(path), 'wb') as stream:
                stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(16000)
                stream.writeframes(b'\x01\x01' * 16000)
            validate_wav(path)
            decoded = decode_audio(str(path), sampling_rate=16000)
        valid = decoded.shape == (16000,) and np.isfinite(decoded).all()
        report['pcm_decoder'] = {'generated_non_speech': True, 'samples': len(decoded), 'finite': bool(np.isfinite(decoded).all())}
        ready = valid and report['pyav_import'] == 'OK' and 'int8' in report['cpu_compute_types']
        report['status'] = 'PACKAGES_AND_DECODER_READY_MICROPHONE_PENDING' if ready else 'BLOCKED_CPU_OR_DECODER'
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        report.update(status='BLOCKED_OPTIONAL_ASR_PREPARATION', failure_type=type(exc).__name__)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-packages', action='store_true')
    args = parser.parse_args()
    if args.output.exists(): parser.error('Use a new readiness report destination.')
    report = readiness(); write_report(args.output, report)
    print(json.dumps({'status': report['status'], 'real_microphone': report['windows_microphone'], 'weights': report['whisper_weights']}))
    if args.require_packages and report['status'] != 'PACKAGES_AND_DECODER_READY_MICROPHONE_PENDING': raise SystemExit(1)
