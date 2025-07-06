import { useState, useRef, useEffect } from "react";
import { Link } from "react-router-dom";


const downloadImagingZip = async (captureId) => {
  try {
    const response = await fetch(`https://128.179.187.191:5000/download-capture-zip/${captureId}`);
    if (!response.ok) throw new Error("Failed to fetch ZIP");

    const blob = await response.blob();
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `capture_${captureId}.zip`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(link.href); // cleanup
  } catch (err) {
    console.error("Imaging ZIP download failed:", err);
    alert("Download failed");
  }
};




export default function Demo() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [showPSFModal, setShowPSFModal] = useState(false);
  const [showImagingModal, setShowImagingModal] = useState(false);
  const [showCamera, setShowCamera] = useState(false);
  const [psfName, setPsfName] = useState("");
  const [algorithm, setAlgorithm] = useState("ADMM");
  const [iterations, setIterations] = useState(10);
  const [psfList, setPsfList] = useState([]);
  const [selectedPsf, setSelectedPsf] = useState(""); 
  const [rawLenslessImage, setRawLenslessImage] = useState(null);
  const [reconImage, setReconImage] = useState(null);
  const [enlargedImage, setEnlargedImage] = useState(null);
  const [psfStatus, setPsfStatus] = useState("");
  const [imagingStatus, setImagingStatus] = useState("");
  const [showPsfDropdown, setShowPsfDropdown] = useState(false);
  const [showIterationModal, setShowIterationModal] = useState(false);
  const [newIterations, setNewIterations] = useState(iterations);
  const [useAutoExposure, setUseAutoExposure] = useState(true);
  const [manualExposure, setManualExposure] = useState(1); // default to 1s
  const [showAlgorithmModal, setShowAlgorithmModal] = useState(false);
  const [newAlgorithm, setNewAlgorithm] = useState(algorithm);
  const [captureName, setCaptureName] = useState("");
  const [captureList, setCaptureList] = useState([]);
  const [selectedCapture, setSelectedCapture] = useState("");
  const [showCaptureDropdown, setShowCaptureDropdown] = useState(false); // ✅ add this
  




  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  const [psfImage, setPsfImage] = useState(null);
  const [autocorrImage, setAutocorrImage] = useState(null);
  const handleLoadCapture = async (id) => {
    try {
      const res = await fetch(`https://128.179.187.191:5000/load-capture/${id}`);
      if (!res.ok) throw new Error("Failed to load capture");
      const data = await res.json();
      setRawLenslessImage(`data:image/png;base64,${data.imgCapture}`);
      setReconImage(`data:image/png;base64,${data.imgRecon}`);
      sessionStorage.setItem("latestCaptureId", id); // set for download & rerun
      if (data.psfName) {
        setSelectedPsf(data.psfName);
        sessionStorage.setItem("selectedPsf", data.psfName);
      }

      if (data.imgUpload) {
        const uploadUrl = `data:image/png;base64,${data.imgUpload}`;
        setImagePreview(uploadUrl); // 🔍 for visual preview

        // Also keep File object in case user reruns reconstruction
        const blob = await (await fetch(uploadUrl)).blob();
        const file = new File([blob], "loaded.png", { type: "image/png" });
        setSelectedFile(file); // ✅ Makes sure the backend can rerun with this
      }
    } catch (err) {
      setImagingStatus("❌ Load failed.");
      console.error(err);
    }
  };
  useEffect(() => {
    if (showCamera && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current;
      videoRef.current.play();
      console.log("Video is now playing via useEffect.");
    }
  }, [showCamera]);

  useEffect(() => {
  fetch('https://128.179.187.191:5000/list-psfs')
    .then(res => res.json())
    .then(data => setPsfList(data.psfs || []))
    .catch(err => {
      setPsfList([]);
      console.error('Could not load PSF list:', err);
    });
}, []);

  useEffect(() => {
    fetch('https://128.179.187.191:5000/list-captures')
      .then(res => res.json())
      .then(data => setCaptureList(data.captures || []))
      .catch(err => {
        setCaptureList([]);
        console.error('Could not load capture list:', err);
      });
  }, []);


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

    setPsfStatus("Capturing PSF image...");

    fetch('https://128.179.187.191:5000/run-demo', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ psfName: psfName.trim() }),
    })
      .then((res) => {
        setPsfStatus("Color correcting & computing autocorrelation...");
        return res.json();
      })
      .then((data) => {
        setPsfImage(`data:image/png;base64,${data.psf}`);
        setAutocorrImage(`data:image/png;base64,${data.autocorr}`);
        setPsfStatus("Done.");
        setTimeout(() => setPsfStatus(""), 3000); // nettoie après 3s
      })
      .catch((err) => {
        setPsfStatus("Error during PSF capture.");
        console.error('Demo run failed:', err);
      });
  };




  const handleRunImaging = async () => {
    if (!selectedFile) {
      alert("Please upload or capture an image!");
      return;
    }
    if (!selectedPsf) {
      alert("Please select a PSF!");
      return;
    }

    const formData = new FormData();
    formData.append('iterations', iterations);
    formData.append('psfChosen', selectedPsf);
    formData.append('image', selectedFile);
    formData.append('useAutoExposure', useAutoExposure.toString()); 
    formData.append('manualExposure', manualExposure.toString()); 
    formData.append('algorithm', algorithm);
    formData.append('captureName', captureName.trim());
    try {
      setShowImagingModal(false);
      setImagingStatus("Uploading and displaying image...");
      const response = await fetch('https://128.179.187.191:5000/run-full-imaging', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error("Request failed");
      setImagingStatus("Capturing from camera, please wait...");
      const data = await response.json();
      setRawLenslessImage(`data:image/png;base64,${data.imgCapture}`);
      setReconImage(`data:image/png;base64,${data.imgRecon}`);
      sessionStorage.setItem("latestCaptureId", data.captureId); // ✅ Store capture ID
      setImagingStatus("✅ Imaging complete.");
    } catch (err) {
      setImagingStatus("❌ Imaging failed.");
      console.error(err);
    }
  };

  const handleRerunReconstruction = async () => {
    const captureId = sessionStorage.getItem("latestCaptureId");
    if (!captureId || !selectedPsf) {
      alert("Missing capture ID or PSF.");
      return;
    }
    setShowIterationModal(false);
    try {
      setImagingStatus(`Rerunning with ${newIterations} iterations...`);

      const res = await fetch("https://128.179.187.191:5000/rerun-reconstruction", {
        method: "POST",
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          captureId,
          psfName: selectedPsf,
          iterations: newIterations,
          algorithm
        })
      });

      if (!res.ok) throw new Error("Reconstruction failed");

      const data = await res.json();
      setReconImage(`data:image/png;base64,${data.recon}`);
      setImagingStatus("✅ Reconstruction updated.");
      setIterations(newIterations); // update main iteration value
      setShowIterationModal(false);
    } catch (err) {
      setImagingStatus("❌ Reconstruction failed.");
      console.error(err);
    }
  };




  const triggerUpload = () => {
    fileInputRef.current.click();
  };

  const triggerCamera = async () => {
    console.log("Requesting webcam access...");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      console.log("Webcam stream obtained:", stream);
      streamRef.current = stream;
      setShowCamera(true);
    } catch (err) {
      console.error("Error accessing webcam: ", err);
    }
  };

  const takeSnapshot = () => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!video || !canvas) {
          console.error("Video or canvas not available for snapshot.");
          return;
        }
        const context = canvas.getContext("2d");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        const dataURL = canvas.toDataURL("image/png");
        setImagePreview(dataURL);
        setShowCamera(false);

        // ✅ Convert dataURL to File and set selectedFile
        fetch(dataURL)
          .then(res => res.blob())
          .then(blob => {
            const file = new File([blob], "captured.png", { type: "image/png" });
            setSelectedFile(file);
          });

        // ✅ Stop the webcam
        if (video.srcObject) {
          video.srcObject.getTracks().forEach((track) => track.stop());
        }
      };

      const handleRetrieveFromBackend = async () => {
        try {
          const res = await fetch('https://128.179.187.191:5000/latest-photo');
          if (!res.ok) throw new Error("Failed to retrieve photo");

          const blob = await res.blob();
          const file = new File([blob], "phone_capture.png", { type: "image/png" });
          setSelectedFile(file);

          const reader = new FileReader();
          reader.onloadend = () => {
            setImagePreview(reader.result);
          };
          reader.readAsDataURL(file);

          setShowCamera(false);  // just in case camera was active
        } catch (err) {
          alert("Could not retrieve selfie from phone.");
          console.error(err);
        }
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
            {psfStatus && (
              <p className="text-green-400 font-mono mb-4 animate-pulse">{psfStatus}</p>
            )}

            <a href="/tutorial/psf-box-setup" className="underline text-green-300">Link to tutorial</a> for setting / placing PSF box
          </p>
          <div className="flex flex-col md:flex-row md:items-center gap-4 mb-6">
            <button onClick={handleCapture} className="bg-gray-200 text-black py-2 px-6 rounded-full hover:bg-gray-300 transition w-full md:w-auto">Capture</button>  
              <button
                className="bg-gray-200 text-black py-2 px-6 rounded-full hover:bg-gray-300 transition w-full md:w-auto"
                onClick={() => {
                  if (selectedPsf) {
                    const downloadZip = async (url, filename) => {
                      try {
                        const response = await fetch(url);
                        if (!response.ok) throw new Error("Failed to fetch ZIP");

                        const blob = await response.blob();
                        const link = document.createElement('a');
                        link.href = URL.createObjectURL(blob);
                        link.download = filename;
                        link.click();
                        URL.revokeObjectURL(link.href); // cleanup
                      } catch (err) {
                        console.error("Download failed:", err);
                        alert("Download failed");
                      }
                    };
                  downloadZip(`https://128.179.187.191:5000/download-psf-zip/${selectedPsf}`, `${selectedPsf}.zip`);
                  }else {
                    alert("Please name your PSF first.");
                  }
                }}
                disabled={!psfImage && !autocorrImage}
              >
                Download
              </button>
              <button
                onClick={() => setShowPsfDropdown(prev => !prev)}
                className="bg-gray-200 text-black py-2 px-6 rounded-full hover:bg-gray-300 transition w-full md:w-auto"
              >
                Load
              </button>
              {showPsfDropdown && (
                <select
                  value={selectedPsf}
                  onChange={async (e) => {
                    const selected = e.target.value;
                    setSelectedPsf(selected);

                    // ✅ Fetch PSF and autocorr images from backend
                    if (selected) {
                      try {
                        const res = await fetch(`https://128.179.187.191:5000/load-psf/${selected}`);
                        if (!res.ok) throw new Error("Could not load selected PSF");
                        const data = await res.json();
                        setPsfImage(`data:image/png;base64,${data.psf}`);
                        setAutocorrImage(`data:image/png;base64,${data.autocorr}`);
                      } catch (err) {
                        console.error("Failed to load PSF images:", err);
                      }
                    }
                  }}

                  className="bg-gray-800 text-green-300 px-4 py-2 rounded w-full md:w-auto"
                >
                  <option value="">-- Select a PSF --</option>
                  {psfList.map((psf) => (
                    <option key={psf} value={psf}>{psf}</option>
                  ))}
                </select>
              )}
          </div>
          <div className="flex flex-col md:flex-row gap-8 mb-6 justify-center items-center">
            {/* PSF */}
            <div className="flex flex-col items-center w-full md:w-1/2">
              {!psfImage ? (
                <div
                  className="flex items-center justify-center w-full h-72 rounded-2xl border-2 border-green-400"
                  style={{
                    background: "linear-gradient(135deg, #1ca7ec66 0%, #1f2f8766 100%)"
                  }}
                >
                  <span className="text-white font-bold text-10l">PSF</span>
                </div>
              ) : (
                <img
                  src={psfImage}
                  alt="PSF"
                  className="object-contain"
                  onClick={() => setEnlargedImage(psfImage)}
                  style={{
                    cursor: 'zoom-in',
                    maxWidth: 500,
                    maxHeight: 500,
                    width: "100%",
                    borderRadius: "1.5rem",
                    boxShadow: "0 4px 24px #0008",
                    background: "transparent",
                    border: "2px solid #00ff99",
                    padding: "0.75rem"
                  }}
                />
              )}
            </div>

            {/* Autocorrelation */}
            <div className="flex flex-col items-center w-full md:w-1/2">
              {!autocorrImage ? (
                <div
                  className="flex items-center justify-center w-full h-72 rounded-2xl border-2 border-green-400"
                  style={{
                    background: "linear-gradient(135deg, #1ca7ec66 0%, #1f2f8766 100%)"
                  }}
                >
                  <span className="text-white font-bold text-10l">Autocorrelation</span>
                </div>
              ) : (
                <img
                  src={autocorrImage}
                  alt="Autocorrelation"
                  className="object-contain"
                  onClick={() => setEnlargedImage(autocorrImage)}
                  style={{
                    cursor: 'zoom-in',
                    maxWidth: 500,
                    maxHeight: 500,
                    width: "100%",
                    borderRadius: "1.5rem",
                    boxShadow: "0 4px 24px #0008",
                    background: "transparent",
                    border: "2px solid #00ff99",
                    padding: "0.75rem"
                  }}
                />
              )}
            </div>
          </div>

        </div>

        <div className="max-w-4xl mx-auto bg-[#1e1b3a] p-8 rounded-2xl border border-green-500 shadow-2xl">
          <h2 className="text-2xl font-semibold text-green-300 mb-4">Lensless Imaging</h2>    
            {imagingStatus && (
              <p className="text-green-400 font-mono animate-pulse md:basis-full mb-4">
                  {imagingStatus}
                </p>
            )}
          <div className="flex flex-col md:flex-row md:items-center gap-4 mb-6">

            <button onClick={handleTakePicture} className="bg-gray-200 text-black py-2 px-6 rounded-full hover:bg-gray-300 transition w-full md:w-auto">Take Picture</button>
            <button
              className="bg-gray-200 text-black py-2 px-6 rounded-full hover:bg-gray-300 transition w-full md:w-auto"
              onClick={() => {
                if (rawLenslessImage && reconImage) {
                  const latestCaptureId = sessionStorage.getItem("latestCaptureId");
                  if (latestCaptureId) {
                    downloadImagingZip(latestCaptureId);
                  } else {
                    alert("No capture ID found for this session.");
                  }
                } else {
                  alert("Nothing to download yet.");
                }
              }}
              disabled={!rawLenslessImage && !reconImage}
            >
              Download
            </button>
            <button
              onClick={() => setShowCaptureDropdown(prev => !prev)}
              className="bg-gray-200 text-black py-2 px-6 rounded-full hover:bg-gray-300 transition w-full md:w-auto"
            >
              Load
            </button>

            {showCaptureDropdown && (
              <select
                value={selectedCapture}
                onChange={(e) => {
                  const id = e.target.value;
                  setSelectedCapture(id);
                  handleLoadCapture(id);
                }}
                className="bg-gray-800 text-green-300 px-4 py-2 rounded w-full md:w-auto"
              >
                <option value="">-- Select a capture --</option>
                {captureList.map((cap) => (
                  <option key={cap} value={cap}>{cap}</option>
                ))}
              </select>
            )}
          </div>

          <div className="flex flex-col md:flex-row gap-8 mb-6 justify-center items-center">
            {/* Raw Lensless */}
            <div className="flex flex-col items-center w-full md:w-1/2">
              {!rawLenslessImage ? (
                <div
                  className="flex items-center justify-center w-full h-72 rounded-2xl border-2 border-green-400"
                  style={{
                    background: "linear-gradient(135deg, #1ca7ec66 0%, #1f2f8766 100%)"
                  }}
                >
                  <span className="text-white font-bold text-10l">Raw Lensless</span>
                </div>
              ) : (
                <img
                  src={rawLenslessImage}
                  alt="Raw Lensless"
                  className="object-contain"
                  onClick={() => setEnlargedImage(rawLenslessImage)}
                  style={{
                    cursor: 'zoom-in',
                    maxWidth: 500,
                    maxHeight: 500,
                    width: "100%",
                    borderRadius: "1.5rem",
                    boxShadow: "0 4px 24px #0008",
                    background: "transparent",
                    border: "2px solid #00ff99",
                    padding: "0.75rem"
                  }}
                />
              )}
            </div>

            {/* Reconstruction */}
            <div className="flex flex-col items-center w-full md:w-1/2">
              {!reconImage ? (
                <div
                  className="flex items-center justify-center w-full h-72 rounded-2xl border-2 border-green-400"
                  style={{
                    background: "linear-gradient(135deg, #1ca7ec66 0%, #1f2f8766 100%)"
                  }}
                >
                  <span className="text-white font-bold text-10l">Reconstruction</span>
                </div>
              ) : (
                <img
                  src={reconImage}
                  alt="Reconstruction"
                  className="object-contain"
                  onClick={() => setEnlargedImage(reconImage)}
                  style={{
                    cursor: 'zoom-in',
                    maxWidth: 500,
                    maxHeight: 500,
                    width: "100%",
                    borderRadius: "1.5rem",
                    boxShadow: "0 4px 24px #0008",
                    background: "transparent",
                    border: "2px solid #00ff99",
                    padding: "0.75rem"
                  }}
                />
              )}
            </div>
          </div>
          {rawLenslessImage && reconImage && (
            <div className="mt-4 flex justify-center gap-4">
              <button
                className="bg-gray-100 text-black px-6 py-2 rounded-full hover:bg-gray-200"
                onClick={() => setShowIterationModal(true)}
              >
                Change Iterations
              </button>

              <button
                className="bg-gray-100 text-black px-6 py-2 rounded-full hover:bg-gray-200"
                onClick={() => setShowAlgorithmModal(true)}
              >
                Change Algorithm
              </button>
            </div>
          )}


          {imagePreview && (
            <div className="text-center">
              <p className="mb-2 text-sm text-gray-400 mt-2">Image Preview:</p>
              <img src={imagePreview} alt="Preview" className="mx-auto max-h-72 border rounded-lg shadow-lg" />
            </div>
          )}
        </div>
      </div>

      {showPSFModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-40">
          <div className="bg-[#2a2a3b] p-8 rounded-2xl w-96 text-white">
            <h3 className="text-xl font-bold mb-4">Name your PSF</h3>
            <p className="text-green-400 mb-4 text-sm ">
              Please include diameter, e.g. <span className="font-semibold">psf_1mm</span>
            </p>
            <input type="text" value={psfName} onChange={(e) => setPsfName(e.target.value)} placeholder="Enter PSF name" className="w-full px-4 py-2 rounded text-black mb-4" />
            <div className="flex gap-4">
              <button onClick={handleRunCapture} className="bg-green-400 text-black px-6 py-2 rounded-full w-full hover:bg-green-300 transition">Save & Capture</button>
              <button onClick={() => setShowPSFModal(false)} className="bg-gray-500 text-white px-6 py-2 rounded-full w-full hover:bg-gray-400 transition">Cancel</button>
            </div>
          </div>
        </div>
      )}

      {showAlgorithmModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-40">
          <div className="bg-[#2a2a3b] p-6 rounded-xl w-96 text-white text-center">
            <h3 className="text-xl font-bold mb-4">Change Algorithm</h3>
            <div className="flex gap-4 mb-4 justify-center">
              <button
                className={`px-4 py-2 rounded ${newAlgorithm === "ADMM" ? "bg-gray-500 text-white" : "bg-gray-200 text-black"}`}
                onClick={() => setNewAlgorithm("ADMM")}
              >
                ADMM
              </button>
              <button
                className={`px-4 py-2 rounded ${newAlgorithm === "Gradient Descent" ? "bg-gray-500 text-white" : "bg-gray-200 text-black"}`}
                onClick={() => setNewAlgorithm("Gradient Descent")}
              >
                Gradient Descent
              </button>
            </div>
            <div className="flex gap-4">
              <button
                onClick={async () => {
                  const captureId = sessionStorage.getItem("latestCaptureId");
                  if (!captureId || !selectedPsf) {
                    alert("Missing capture ID or PSF.");
                    return;
                  }
                  setShowAlgorithmModal(false);

                  try {
                    setImagingStatus(`Reconstructing using ${newAlgorithm}...`);
                    const res = await fetch("https://128.179.187.191:5000/rerun-reconstruction", {
                      method: "POST",
                      headers: { 'Content-Type': 'application/json' },
                      body: JSON.stringify({
                        captureId,
                        psfName: selectedPsf,
                        iterations,
                        algorithm: newAlgorithm
                      })
                    });

                    if (!res.ok) throw new Error("Reconstruction failed");
                    const data = await res.json();
                    setReconImage(`data:image/png;base64,${data.recon}`);
                    setImagingStatus(`✅ Reconstructed using ${newAlgorithm}.`);
                    setAlgorithm(newAlgorithm);
                  } catch (err) {
                    console.error(err);
                    setImagingStatus("❌ Reconstruction failed.");
                  }
                }}
                className="bg-green-400 text-black px-6 py-2 rounded-full w-full hover:bg-green-300"
              >
                Rerun
              </button>
              <button
                onClick={() => setShowAlgorithmModal(false)}
                className="bg-gray-500 text-white px-6 py-2 rounded-full w-full hover:bg-gray-400"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {showImagingModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-40">
          <div className="bg-[#2a2a3b] p-8 rounded-2xl w-96 text-white">
            <h3 className="text-xl font-bold mb-4">Taking a Picture</h3>
            <div className="mb-6">
              <label htmlFor="psf-dropdown" className="block mb-2 text-gray-400">
                Select a PSF:
              </label>
              <select
                id="psf-dropdown"
                value={selectedPsf}
                onChange={e => setSelectedPsf(e.target.value)}
                className="px-4 py-2 rounded bg-gray-900 text-green-300 w-full md:w-auto"
              >
                <option value="">-- Select a PSF --</option>
                {psfList.map(name => (
                  <option key={name} value={name}>{name}</option>
                ))}
              </select>
              {!selectedPsf && <p className="text-red-400 mt-2 text-sm">Please select a PSF above.</p>}
            </div>

            
            <div className="flex gap-4 mb-6">
              <button onClick={triggerUpload}
                      className="text-sm bg-gray-100 text-gray-900  
                                px-6 py-3 rounded-lg  
                                hover:bg-gray-200  
                                focus:outline-none focus:ring-2 focus:ring-green-400  
                                transition-colors duration-200">
                Upload
              </button>
              <button onClick={triggerCamera}
                      className="text-sm bg-gray-100 text-gray-900 px-6 py-3 rounded-lg hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-green-400 transition-colors duration-200">
                From your device
              </button>
              <button onClick={handleRetrieveFromBackend}
                      className="text-sm bg-gray-100 text-gray-900 px-6 py-3 rounded-lg hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-green-400 transition-colors duration-200">
                Retrieve from Phone
              </button>
            </div>
            <p className="mb-2 text-sm">Algorithm parameters</p>
                          <div className="flex items-center gap-4 mb-4">
                <label className="flex items-center">
                  <input type="radio" checked={useAutoExposure} onChange={() => setUseAutoExposure(true)} className="mr-2" />
                  Auto Exposure
                </label>
                <label className="flex items-center">
                  <input type="radio" checked={!useAutoExposure} onChange={() => setUseAutoExposure(false)} className="mr-2" />
                  Manual Exposure
                </label>
              </div>

              {!useAutoExposure && (
                <div className="mb-4">
                  <label className="block mb-1">Set Exposure :</label>
                  <input
                    type="number"
                    step="any"
                    value={manualExposure}
                    min={0.001}
                    onChange={(e) => setManualExposure(Number(e.target.value))}
                    className="w-full px-3 py-1 rounded text-black"
                  />
                </div>
              )}
            <div className="flex gap-4 mb-4">
              <button onClick={() => setAlgorithm("ADMM")} className={`px-4 py-2 rounded ${algorithm === "ADMM" ? "bg-gray-500 text-white" : "bg-gray-200 text-black"}`}>ADMM</button>
              <button onClick={() => setAlgorithm("Gradient Descent")} className={`px-4 py-2 rounded ${algorithm === "Gradient Descent" ? "bg-gray-500 text-white" : "bg-gray-200 text-black"}`}>Gradient descent</button>
            </div>
            <label className="block mb-2">Number of iterations: {iterations}</label>
            <input type="range" min="1" max="100" value={iterations} onChange={(e) => setIterations(Number(e.target.value))} className="w-full" />
            <div className="mb-4">
              <label className="block mb-2 text-sm text-gray-300">Image folder name :</label>
              <input
                type="text"
                value={captureName}
                onChange={(e) => setCaptureName(e.target.value)}
                placeholder="Enter folder name"
                className="w-full px-4 py-2 rounded text-black"
              />
            </div>
            <button
              onClick={handleRunImaging}
              disabled={!selectedPsf || !selectedFile}
              className={`mt-6 bg-white text-black w-full py-2 rounded-full hover:bg-gray-300 ${!selectedPsf ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              Take picture!
            </button>
            <button
              onClick={() => setShowImagingModal(false)}
              className="mt-2 bg-gray-500 text-white w-full py-2 rounded-full hover:bg-gray-400"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {showIterationModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-40">
          <div className="bg-[#2a2a3b] p-6 rounded-xl w-96 text-white text-center">
            <h3 className="text-xl font-bold mb-4">Change Iterations</h3>
            <input
              type="number"
              min="1"
              max="100"
              value={newIterations}
              onChange={(e) => setNewIterations(Number(e.target.value))}
              className="w-full px-4 py-2 mb-4 rounded text-black"
            />
            <div className="flex gap-4">
              <button
                onClick={handleRerunReconstruction}
                className="bg-green-400 text-black px-6 py-2 rounded-full w-full hover:bg-green-300"
              >
                Rerun Reconstruction
              </button>
              <button
                onClick={() => setShowIterationModal(false)}
                className="bg-gray-500 text-white px-6 py-2 rounded-full w-full hover:bg-gray-400"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}


      {showCamera && (
        <div className="fixed inset-0 bg-black bg-opacity-80 flex justify-center items-center z-50">
          <div className="bg-[#1f1f2f] p-6 rounded-2xl text-white text-center">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              className="w-full rounded mb-4 bg-black"
              style={{ minHeight: "200px" }}
            />
            <button onClick={takeSnapshot} className="bg-green-400 px-6 py-2 rounded-full text-black font-semibold hover:bg-green-300">Capture Snapshot</button>
            <canvas ref={canvasRef} className="hidden"></canvas>
          </div>
        </div>
      )}
      {enlargedImage && (
        <div
          className="fixed inset-0 bg-black bg-opacity-80 flex justify-center items-center z-50"
          onClick={() => setEnlargedImage(null)}
        >
          <div
            className="relative"
            onClick={e => e.stopPropagation()}
          >
            <button
              onClick={() => setEnlargedImage(null)}
              className="absolute top-2 right-2 bg-black bg-opacity-60 rounded-full px-4 py-2 text-white text-lg font-bold z-10 hover:bg-opacity-90"
              aria-label="Close"
            >
              ×
            </button>
            <img
              src={enlargedImage}
              alt="Enlarged"
              className="max-w-[90vw] max-h-[80vh] rounded-2xl shadow-2xl border-4 border-white"
            />
          </div>
        </div>
      )}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        className="hidden"
      />
    </div>
  );
}
