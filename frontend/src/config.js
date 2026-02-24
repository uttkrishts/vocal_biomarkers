/**
 * Centralized API configuration.
 *
 * All URLs are derived from a single VITE_API_BASE_URL env variable.
 * In development, defaults to http://localhost:8000.
 * Set VITE_API_BASE_URL in .env (or .env.local) for production/staging.
 */

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/+$/, '');

// Derive WebSocket URL from the HTTP base (http → ws, https → wss)
const WS_BASE_URL = API_BASE_URL.replace(/^http/, 'ws');

const config = {
    /** Base URL for REST API calls */
    API_BASE_URL,

    /** Base URL for WebSocket connections */
    WS_BASE_URL,

    /** REST endpoints */
    endpoints: {
        health: `${API_BASE_URL}/health`,
        analyzeFile: `${API_BASE_URL}/analyze/file`,
    },

    /** WebSocket endpoints */
    ws: {
        analyzeStream: `${WS_BASE_URL}/analyze/stream`,
    },

    /** Supported audio file extensions */
    ALLOWED_EXTENSIONS: ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.webm'],
};

export default config;
