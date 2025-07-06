// src/PhoneCapturePage.tsx
import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';

export default function PhoneCapturePage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [status, setStatus] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    async function initCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            videoRef.current?.play().catch(err => {
              console.error("Video play() failed:", err);
            });
          };
        }

      } catch (err) {
        console.error("Camera access failed", err);
        setStatus("Camera access denied.");
      }
    }
    initCamera();

    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  const handleCaptureAndUpload = async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video || !canvas) return;

    const ctx = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx?.drawImage(video, 0, 0, canvas.width, canvas.height);

    const dataUrl = canvas.toDataURL('image/png');
    const blob = await (await fetch(dataUrl)).blob();
    const file = new File([blob], 'selfie.png', { type: 'image/png' });

    const formData = new FormData();
    formData.append('photo', file);

    try {
      setStatus("Uploading...");
      const res = await fetch('https://128.179.187.191:5000/upload-photo', {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) throw new Error("Upload failed");
      setStatus("✅ Uploaded successfully");
      setTimeout(() => navigate('/demo'), 2000);
    } catch (err) {
      console.error(err);
      setStatus("❌ Upload failed");
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-[#1a1a2e] text-white p-8">
      <h1 className="text-2xl font-bold mb-6">Take a Selfie for Lensless Imaging</h1>
      <video
        ref={videoRef}
        autoPlay
        playsInline
        className="rounded-lg w-full max-w-md mb-4 bg-black"
        style={{
          minHeight: "200px",
          transform: "scaleX(-1)" // ✅ Mirror the video
        }}
      />
      <canvas ref={canvasRef} className="hidden" />
      <div className="flex gap-4">
        <button
          onClick={handleCaptureAndUpload}
          className="bg-green-400 px-6 py-2 rounded-full text-black font-semibold hover:bg-green-300"
        >
          Capture & Upload
        </button>
        <button
          onClick={() => navigate('/demo')}
          className="bg-gray-500 px-6 py-2 rounded-full text-white hover:bg-gray-400"
        >
          Cancel
        </button>
      </div>
      {status && <p className="mt-4 text-green-300 font-mono animate-pulse">{status}</p>}
    </div>
  );
}
