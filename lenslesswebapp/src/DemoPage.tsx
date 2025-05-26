import { useState, useRef, useEffect } from "react";
import { Link } from "react-router-dom";

export default function Demo() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [showPSFModal, setShowPSFModal] = useState(false);
  const [showImagingModal, setShowImagingModal] = useState(false);
  const [showCamera, setShowCamera] = useState(false);
  const [psfName, setPsfName] = useState("");
  const [algorithm, setAlgorithm] = useState("ADMM");
  const [iterations, setIterations] = useState(10);

  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const [psfImage, setPsfImage] = useState(null);
  const [autocorrImage, setAutocorrImage] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setSelectedFile(file);
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleCapture = () => {
    setShowPSFModal(true);
  };

  const handleTakePicture = () => {
    setShowImagingModal(true);
  };

  const handleRunCapture = () => {
    setShowPSFModal(false);
    fetch('http://128.179.187.191:5000/run-demo', {
      method: 'POST',
    })
      .then((res) => res.json())
      .then((data) => {
        setPsfImage(`data:image/png;base64,${data.psf}`);
        setAutocorrImage(`data:image/png;base64,${data.autocorr}`);
      })
      .catch((err) => {
        console.error('Demo run failed:', err);
      });
  };

  const handleRunImaging = () => {
    alert(`Running ${algorithm} for ${iterations} iterations`);
    setShowImagingModal(false);
  };

  const triggerUpload = () => {
    fileInputRef.current.click();
  };

  const triggerCamera = async () => {
    setShowCamera(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
    } catch (err) {
      console.error("Error accessing webcam: ", err);
    }
  };

  const takeSnapshot = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataURL = canvas.toDataURL("image/png");
    setImagePreview(dataURL);
    setShowCamera(false);
    video.srcObject.getTracks().forEach((track) => track.stop());
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#0a0026] via-[#001f4d] to-[#001322] text-white">
      <nav className="sticky top-0 z-30 bg-[#0f0c29]/80 backdrop-blur-md border-b border-green-400 shadow-lg flex items-center justify-between p-6">
        <Link to="/">
          <img src="/logo.png" alt="Lensless Logo" className="w-12 h-12 object-contain hover:scale-110 transition-transform duration-300" />
        </Link>
        <ul className="flex gap-6 text-sm text-gray-300">
          <li><Link to="/" className="hover:text-green-300 transition">Home</Link></li>
          <li><a href="https://github.com/LCAV/LenslessPiCam" target="_blank" className="hover:text-green-300 transition" rel="noreferrer">GitHub</a></li>
          <li><a href="mailto:info@lenslesspicam.com" className="hover:text-green-300 transition">Contact</a></li>
        </ul>
      </nav>

      <div className="p-10">
        <h1 className="text-4xl font-bold mb-10 text-green-300 text-center">Lensless Imaging Demo</h1>

        <div className="max-w-4xl mx-auto bg-[#1e1b3a] p-8 rounded-2xl border border-green-500 shadow-2xl mb-10">
          <h2 className="text-2xl font-semibold text-green-300 mb-2">PSF Calibration</h2>
          <p className="text-gray-400 mb-6">
            <a href="/tutorial/psf-box-setup" className="underline text-green-300">Link to tutorial</a> for setting / placing PSF box
          </p>
          <div className="flex flex-col md:flex-row md:items-center gap-4 mb-6">
            <button onClick={handleCapture} className="bg-gray-200 text-black py-2 px-6 rounded-full hover:bg-gray-300 transition w-full md:w-auto">Capture</button>
            <button className="bg-gray-200 text-black py-2 px-6 rounded-full hover:bg-gray-300 transition w-full md:w-auto">Download</button>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-blue-700 h-40 rounded-xl flex items-center justify-center">
              {psfImage ? (
                <img src={psfImage} alt="PSF" className="h-full rounded-xl object-contain" />
              ) : (
                <span className="text-white font-semibold text-xl">PSF</span>
              )}
            </div>
            <div className="bg-blue-700 h-40 rounded-xl flex items-center justify-center">
              {autocorrImage ? (
                <img src={autocorrImage} alt="Autocorrelation" className="h-full rounded-xl object-contain" />
              ) : (
                <span className="text-white font-semibold text-xl">Autocorrelations</span>
              )}
            </div>
          </div>
        </div>

        <div className="max-w-4xl mx-auto bg-[#1e1b3a] p-8 rounded-2xl border border-green-500 shadow-2xl">
          <h2 className="text-2xl font-semibold text-green-300 mb-2">Imaging</h2>
          <p className="text-gray-400 mb-6">
            Dropdown menu of different PSFs available on the Raspberry Pi (first time use → no PSF available)
          </p>

          <div className="flex flex-col md:flex-row md:items-center gap-4 mb-6">
            <button onClick={handleTakePicture} className="bg-gray-200 text-black py-2 px-6 rounded-full hover:bg-gray-300 transition w-full md:w-auto">Take Picture</button>
            <button className="bg-gray-200 text-black py-2 px-6 rounded-full hover:bg-gray-300 transition w-full md:w-auto">Download</button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <div className="bg-blue-700 h-40 rounded-xl flex items-center justify-center text-white font-semibold text-xl">Raw Lensless</div>
            <div className="bg-blue-700 h-40 rounded-xl flex items-center justify-center text-white font-semibold text-xl">Reconstruction</div>
          </div>

          {imagePreview && (
            <div className="text-center">
              <p className="mb-2 text-sm text-gray-400">Image Preview:</p>
              <img src={imagePreview} alt="Preview" className="mx-auto max-h-72 border rounded-lg shadow-lg" />
            </div>
          )}
        </div>
      </div>

      {showPSFModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-40">
          <div className="bg-[#2a2a3b] p-8 rounded-2xl w-96 text-white">
            <h3 className="text-xl font-bold mb-4">Name your PSF</h3>
            <input type="text" value={psfName} onChange={(e) => setPsfName(e.target.value)} placeholder="Enter PSF name" className="w-full px-4 py-2 rounded text-black mb-4" />
            <button onClick={handleRunCapture} className="bg-green-400 text-black px-6 py-2 rounded-full w-full hover:bg-green-300 transition">Save & Capture</button>
          </div>
        </div>
      )}

      {showImagingModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-40">
          <div className="bg-[#2a2a3b] p-8 rounded-2xl w-96 text-white">
            <h3 className="text-xl font-bold mb-4">Taking a Picture</h3>
            <div className="flex justify-between mb-4">
              <button onClick={triggerUpload} className="bg-gray-200 text-black px-4 py-2 rounded">Upload</button>
              <span className="text-gray-300 self-center">or</span>
              <button onClick={triggerCamera} className="bg-gray-200 text-black px-4 py-2 rounded">From your device</button>
              <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={handleFileChange} />
            </div>
            <p className="mb-2 text-sm">Algorithm parameters</p>
            <div className="flex gap-4 mb-4">
              <button onClick={() => setAlgorithm("ADMM")} className={`px-4 py-2 rounded ${algorithm === "ADMM" ? "bg-gray-500 text-white" : "bg-gray-200 text-black"}`}>ADMM</button>
              <button onClick={() => setAlgorithm("Gradient Descent")} className={`px-4 py-2 rounded ${algorithm === "Gradient Descent" ? "bg-gray-500 text-white" : "bg-gray-200 text-black"}`}>Gradient descent</button>
            </div>
            <label className="block mb-2">Number of iterations: {iterations}</label>
            <input type="range" min="1" max="100" value={iterations} onChange={(e) => setIterations(e.target.value)} className="w-full" />
            <button onClick={handleRunImaging} className="mt-6 bg-white text-black w-full py-2 rounded-full hover:bg-gray-300">Take picture!</button>
          </div>
        </div>
      )}

      {showCamera && (
        <div className="fixed inset-0 bg-black bg-opacity-80 flex justify-center items-center z-50">
          <div className="bg-[#1f1f2f] p-6 rounded-2xl text-white text-center">
            <video ref={videoRef} autoPlay playsInline className="w-full rounded mb-4" />
            <button onClick={takeSnapshot} className="bg-green-400 px-6 py-2 rounded-full text-black font-semibold hover:bg-green-300">Capture Snapshot</button>
            <canvas ref={canvasRef} className="hidden"></canvas>
          </div>
        </div>
      )}
    </div>
  );
}
