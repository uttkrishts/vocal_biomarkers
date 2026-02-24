import { useRef, useEffect } from 'react';

export default function WaveformVisualizer({ analyserNode, isActive }) {
    const canvasRef = useRef(null);
    const animFrameRef = useRef(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || !analyserNode || !isActive) {
            // Draw a flat line when inactive
            if (canvas) {
                const ctx = canvas.getContext('2d');
                const w = canvas.width;
                const h = canvas.height;
                ctx.clearRect(0, 0, w, h);
                ctx.strokeStyle = 'rgba(45, 212, 191, 0.3)';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(0, h / 2);
                ctx.lineTo(w, h / 2);
                ctx.stroke();
            }
            return;
        }

        const ctx = canvas.getContext('2d');
        const bufferLength = analyserNode.fftSize;
        const dataArray = new Float32Array(bufferLength);

        const draw = () => {
            animFrameRef.current = requestAnimationFrame(draw);
            analyserNode.getFloatTimeDomainData(dataArray);

            const w = canvas.width;
            const h = canvas.height;

            ctx.clearRect(0, 0, w, h);

            // Gradient stroke
            const grad = ctx.createLinearGradient(0, 0, w, 0);
            grad.addColorStop(0, '#2dd4bf');
            grad.addColorStop(0.5, '#a78bfa');
            grad.addColorStop(1, '#fb7185');

            ctx.strokeStyle = grad;
            ctx.lineWidth = 2;
            ctx.beginPath();

            const sliceWidth = w / bufferLength;
            let x = 0;

            for (let i = 0; i < bufferLength; i++) {
                const v = dataArray[i];
                const y = (v * 0.5 + 0.5) * h;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
                x += sliceWidth;
            }

            ctx.lineTo(w, h / 2);
            ctx.stroke();
        };

        draw();

        return () => {
            if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
        };
    }, [analyserNode, isActive]);

    // Resize canvas to match container
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const resize = () => {
            const rect = canvas.parentElement.getBoundingClientRect();
            canvas.width = rect.width;
            canvas.height = rect.height;
        };

        resize();
        window.addEventListener('resize', resize);
        return () => window.removeEventListener('resize', resize);
    }, []);

    return (
        <div className="waveform-container">
            <canvas ref={canvasRef} className="waveform-canvas" />
        </div>
    );
}
