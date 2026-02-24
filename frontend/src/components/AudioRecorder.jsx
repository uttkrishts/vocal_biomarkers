import { useState, useRef, useCallback, useEffect } from 'react';
import WaveformVisualizer from './WaveformVisualizer';
import config from '../config';

export default function AudioRecorder({ onResult, onError, onLoading }) {
    const [recording, setRecording] = useState(false);
    const [duration, setDuration] = useState(0);
    const [hasRecording, setHasRecording] = useState(false);
    const [analyserNode, setAnalyserNode] = useState(null);

    const mediaRecorderRef = useRef(null);
    const audioChunksRef = useRef([]);
    const timerRef = useRef(null);
    const streamRef = useRef(null);
    const audioCtxRef = useRef(null);
    const wsRef = useRef(null);

    // Clean up on unmount
    useEffect(() => {
        return () => {
            stopRecording();
            if (wsRef.current) wsRef.current.close();
        };
    }, []);

    const formatTime = (s) => {
        const m = Math.floor(s / 60);
        const sec = s % 60;
        return `${m}:${sec.toString().padStart(2, '0')}`;
    };

    const startRecording = useCallback(async () => {
        try {
            onResult(null);
            onError(null);
            audioChunksRef.current = [];
            setDuration(0);
            setHasRecording(false);

            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    channelCount: 1,
                    sampleRate: 16000,
                    echoCancellation: true,
                    noiseSuppression: true,
                },
            });
            streamRef.current = stream;

            // Set up analyser for waveform
            const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            audioCtxRef.current = audioCtx;
            const source = audioCtx.createMediaStreamSource(stream);
            const analyser = audioCtx.createAnalyser();
            analyser.fftSize = 2048;
            source.connect(analyser);
            setAnalyserNode(analyser);

            // Create MediaRecorder
            const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                ? 'audio/webm;codecs=opus'
                : 'audio/webm';

            const recorder = new MediaRecorder(stream, { mimeType });
            mediaRecorderRef.current = recorder;

            recorder.ondataavailable = (e) => {
                if (e.data.size > 0) {
                    audioChunksRef.current.push(e.data);
                }
            };

            recorder.onstop = () => {
                setHasRecording(audioChunksRef.current.length > 0);
            };

            recorder.start(250); // Collect chunks every 250ms
            setRecording(true);

            // Timer
            timerRef.current = setInterval(() => {
                setDuration((d) => d + 1);
            }, 1000);
        } catch (err) {
            onError('Microphone access denied. Please allow microphone access and try again.');
        }
    }, [onResult, onError]);

    const stopRecording = useCallback(() => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
            mediaRecorderRef.current.stop();
        }
        if (streamRef.current) {
            streamRef.current.getTracks().forEach((t) => t.stop());
            streamRef.current = null;
        }
        if (audioCtxRef.current) {
            audioCtxRef.current.close();
            audioCtxRef.current = null;
        }
        if (timerRef.current) {
            clearInterval(timerRef.current);
            timerRef.current = null;
        }
        setRecording(false);
        setAnalyserNode(null);
    }, []);

    const analyzeRecording = useCallback(async () => {
        if (audioChunksRef.current.length === 0) return;

        onLoading(true);
        onError(null);

        try {
            // Build a blob from the recorded chunks
            const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' });

            // Use WebSocket to stream and analyze
            const ws = new WebSocket(config.ws.analyzeStream);
            wsRef.current = ws;

            ws.onopen = async () => {
                // Send the blob as binary chunks
                const arrayBuffer = await blob.arrayBuffer();
                const chunkSize = 64 * 1024; // 64KB chunks
                for (let i = 0; i < arrayBuffer.byteLength; i += chunkSize) {
                    const chunk = arrayBuffer.slice(i, i + chunkSize);
                    ws.send(chunk);
                }
                // Signal analysis
                ws.send(JSON.stringify({ action: 'analyze', quantize: true }));
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data.error) {
                        onError(data.error);
                    } else {
                        onResult(data);
                    }
                } catch {
                    onError('Invalid response from server');
                }
                onLoading(false);
                ws.close();
            };

            ws.onerror = () => {
                onError('WebSocket connection failed. Is the backend running?');
                onLoading(false);
            };

            ws.onclose = () => {
                wsRef.current = null;
            };
        } catch (err) {
            onError(err.message || 'Analysis failed');
            onLoading(false);
        }
    }, [onResult, onError, onLoading]);

    const resetRecording = useCallback(() => {
        audioChunksRef.current = [];
        setDuration(0);
        setHasRecording(false);
        onResult(null);
        onError(null);
    }, [onResult, onError]);

    return (
        <div className="recorder-section">
            <h3>Record Your Voice</h3>
            <p className="hint">Speak for at least 30 seconds for best results. Single voice, English preferred.</p>

            <WaveformVisualizer analyserNode={analyserNode} isActive={recording} />

            <div className="record-btn-wrapper">
                <button
                    className={`record-btn ${recording ? 'recording' : ''}`}
                    onClick={recording ? stopRecording : startRecording}
                    title={recording ? 'Stop recording' : 'Start recording'}
                >
                    <div className="btn-icon" />
                </button>

                <div className="timer">{formatTime(duration)}</div>
                {recording && <div className="timer-hint">Recording… speak naturally</div>}
                {!recording && hasRecording && (
                    <div className="timer-hint">Recording complete — {formatTime(duration)}</div>
                )}
            </div>

            {!recording && hasRecording && (
                <div className="upload-actions">
                    <button className="analyze-btn" onClick={analyzeRecording}>
                        🔬 Analyze Recording
                    </button>
                    <button
                        className="analyze-btn"
                        onClick={resetRecording}
                        style={{ background: 'rgba(255,255,255,0.08)', color: 'var(--text-secondary)' }}
                    >
                        ↺ Reset
                    </button>
                </div>
            )}
        </div>
    );
}
