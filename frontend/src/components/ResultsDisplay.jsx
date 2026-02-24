const CIRCUMFERENCE = 2 * Math.PI * 50; // r = 50

function severityClass(label) {
    if (!label) return '';
    const l = label.toLowerCase();
    if (l.includes('no ') || l === 'none') return 'severity-none';
    if (l.includes('mild')) return 'severity-mild';
    if (l.includes('moderate')) return 'severity-moderate';
    if (l.includes('severe')) return 'severity-severe';
    return '';
}

function GaugeRing({ score, max, type }) {
    const pct = Math.min(score / max, 1);
    const offset = CIRCUMFERENCE * (1 - pct);

    return (
        <div className="gauge-ring">
            {/* SVG gradients */}
            <svg viewBox="0 0 120 120">
                <defs>
                    <linearGradient id="grad-dep" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stopColor="#60a5fa" />
                        <stop offset="100%" stopColor="#a78bfa" />
                    </linearGradient>
                    <linearGradient id="grad-anx" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stopColor="#fb7185" />
                        <stop offset="100%" stopColor="#fbbf24" />
                    </linearGradient>
                </defs>
                <circle className="track" cx="60" cy="60" r="50" />
                <circle
                    className={`fill ${type}`}
                    cx="60"
                    cy="60"
                    r="50"
                    strokeDasharray={CIRCUMFERENCE}
                    strokeDashoffset={offset}
                />
            </svg>
            <div className="gauge-center">
                <span className="score-value">{score}</span>
                <span className="score-max">/ {max}</span>
            </div>
        </div>
    );
}

export default function ResultsDisplay({ results }) {
    if (!results) return null;

    const { scores, quantized, labels } = results;

    // For quantized: depression max = 2, anxiety max = 3
    const depMax = quantized ? 2 : 1;
    const anxMax = quantized ? 3 : 1;

    const depScore = quantized ? scores.depression : scores.depression;
    const anxScore = quantized ? scores.anxiety : scores.anxiety;

    // Format raw scores for display
    const depDisplay = quantized ? scores.depression : scores.depression.toFixed(3);
    const anxDisplay = quantized ? scores.anxiety : scores.anxiety.toFixed(3);

    return (
        <div className="results-section">
            <div className="results-header">
                <h2>Analysis Results</h2>
                <p>{quantized ? 'Quantized severity levels' : 'Raw model scores'}</p>
            </div>

            <div className="results-grid">
                {/* Depression Card */}
                <div className="score-card depression glass-card">
                    <div className="card-label">Depression</div>
                    {quantized ? (
                        <>
                            <GaugeRing score={depScore} max={depMax} type="depression" />
                            {labels?.depression && (
                                <>
                                    <div className={`severity-label ${severityClass(labels.depression.label)}`}>
                                        {labels.depression.label}
                                    </div>
                                    <div className="severity-description">{labels.depression.description}</div>
                                </>
                            )}
                        </>
                    ) : (
                        <div style={{ textAlign: 'center', padding: '1.5rem 0' }}>
                            <div className="score-value" style={{ fontSize: '2rem', fontWeight: 700 }}>
                                {depDisplay}
                            </div>
                            <div className="severity-description" style={{ marginTop: '0.5rem' }}>
                                Raw score (correlates with PHQ-9)
                            </div>
                        </div>
                    )}
                </div>

                {/* Anxiety Card */}
                <div className="score-card anxiety glass-card">
                    <div className="card-label">Anxiety</div>
                    {quantized ? (
                        <>
                            <GaugeRing score={anxScore} max={anxMax} type="anxiety" />
                            {labels?.anxiety && (
                                <>
                                    <div className={`severity-label ${severityClass(labels.anxiety.label)}`}>
                                        {labels.anxiety.label}
                                    </div>
                                    <div className="severity-description">{labels.anxiety.description}</div>
                                </>
                            )}
                        </>
                    ) : (
                        <div style={{ textAlign: 'center', padding: '1.5rem 0' }}>
                            <div className="score-value" style={{ fontSize: '2rem', fontWeight: 700 }}>
                                {anxDisplay}
                            </div>
                            <div className="severity-description" style={{ marginTop: '0.5rem' }}>
                                Raw score (correlates with GAD-7)
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
