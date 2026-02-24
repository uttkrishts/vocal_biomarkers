import { useState, useRef, useCallback } from 'react';
import config from '../config';

const ALLOWED = config.ALLOWED_EXTENSIONS;

function formatSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function FileUpload({ onResult, onError, onLoading }) {
    const [file, setFile] = useState(null);
    const [dragOver, setDragOver] = useState(false);
    const inputRef = useRef(null);

    const validateFile = useCallback(
        (f) => {
            const ext = '.' + f.name.split('.').pop().toLowerCase();
            if (!ALLOWED.includes(ext)) {
                onError(`Unsupported file type "${ext}". Supported: ${ALLOWED.join(', ')}`);
                return false;
            }
            return true;
        },
        [onError]
    );

    const handleFile = useCallback(
        (f) => {
            if (validateFile(f)) {
                setFile(f);
                onResult(null);
                onError(null);
            }
        },
        [validateFile, onResult, onError]
    );

    const handleDrop = useCallback(
        (e) => {
            e.preventDefault();
            setDragOver(false);
            const f = e.dataTransfer.files?.[0];
            if (f) handleFile(f);
        },
        [handleFile]
    );

    const handleDragOver = (e) => {
        e.preventDefault();
        setDragOver(true);
    };

    const handleDragLeave = () => setDragOver(false);

    const removeFile = () => {
        setFile(null);
        onResult(null);
        if (inputRef.current) inputRef.current.value = '';
    };

    const uploadAndAnalyze = useCallback(async () => {
        if (!file) return;

        onLoading(true);
        onError(null);

        try {
            const formData = new FormData();
            formData.append('file', file);

            const res = await fetch(`${config.endpoints.analyzeFile}?quantize=true`, {
                method: 'POST',
                body: formData,
            });

            const data = await res.json();

            if (!res.ok) {
                throw new Error(data.error || `Server error (${res.status})`);
            }

            onResult(data);
        } catch (err) {
            onError(err.message || 'Upload failed. Is the backend running?');
        } finally {
            onLoading(false);
        }
    }, [file, onResult, onError, onLoading]);

    return (
        <div className="upload-section">
            <h3>Upload Audio File</h3>
            <p className="hint">WAV, MP3, FLAC, OGG, M4A, or WebM — at least 30 seconds of speech recommended.</p>

            <div
                className={`drop-zone ${dragOver ? 'drag-over' : ''}`}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onClick={() => inputRef.current?.click()}
            >
                <span className="upload-icon">🎵</span>
                <div className="upload-text">
                    Drag & drop your audio file here
                    <br />
                    or <strong>click to browse</strong>
                </div>
                <input
                    ref={inputRef}
                    type="file"
                    accept={ALLOWED.join(',')}
                    onChange={(e) => {
                        const f = e.target.files?.[0];
                        if (f) handleFile(f);
                    }}
                />
            </div>

            {file && (
                <>
                    <div className="file-info">
                        <span className="file-icon">🎧</span>
                        <div className="file-details">
                            <div className="file-name">{file.name}</div>
                            <div className="file-size">{formatSize(file.size)}</div>
                        </div>
                        <button className="remove-btn" onClick={removeFile} title="Remove file">
                            ✕
                        </button>
                    </div>

                    <div className="upload-actions">
                        <button className="analyze-btn" onClick={uploadAndAnalyze}>
                            🔬 Analyze File
                        </button>
                    </div>
                </>
            )}
        </div>
    );
}
