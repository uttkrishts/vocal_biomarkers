import { useState } from 'react';
import AudioRecorder from './components/AudioRecorder';
import FileUpload from './components/FileUpload';
import ResultsDisplay from './components/ResultsDisplay';
import './App.css';

export default function App() {
  const [activeTab, setActiveTab] = useState('record');
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  return (
    <>
      <div className="ambient-bg" />

      <div className="app-container">
        {/* Header */}
        <header className="app-header">
          <div className="logo-icon">🧠</div>
          <h1>Vocal Biomarker Analyzer</h1>
          <p>
            Analyze voice recordings for depression &amp; anxiety biomarkers
            <br />
            powered by the KintsugiHealth DAM model
          </p>
        </header>

        {/* Tab Navigation */}
        <nav className="tab-nav">
          <button
            className={`tab-btn ${activeTab === 'record' ? 'active' : ''}`}
            onClick={() => {
              setActiveTab('record');
              setResults(null);
              setError(null);
            }}
          >
            <span className="icon">🎙️</span> Record
          </button>
          <button
            className={`tab-btn ${activeTab === 'upload' ? 'active' : ''}`}
            onClick={() => {
              setActiveTab('upload');
              setResults(null);
              setError(null);
            }}
          >
            <span className="icon">📁</span> Upload
          </button>
        </nav>

        {/* Content */}
        <div className="glass-card">
          {activeTab === 'record' ? (
            <AudioRecorder
              onResult={setResults}
              onError={setError}
              onLoading={setLoading}
            />
          ) : (
            <FileUpload
              onResult={setResults}
              onError={setError}
              onLoading={setLoading}
            />
          )}

          {/* Loading */}
          {loading && (
            <div className="status-banner loading">
              <div className="spinner" />
              Analyzing voice biomarkers — this may take a moment…
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="status-banner error">
              ⚠️ {error}
            </div>
          )}
        </div>

        {/* Results */}
        <ResultsDisplay results={results} />

        {/* Disclaimer */}
        <div className="disclaimer">
          ⚕️ This tool is for research purposes only. It is not intended for diagnosis or
          self-diagnosis without clinical oversight. Results should be interpreted by qualified
          healthcare professionals.
        </div>
      </div>
    </>
  );
}
