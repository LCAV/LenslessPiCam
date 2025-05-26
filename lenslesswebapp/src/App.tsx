import { Link } from "react-router-dom";
import { useEffect, useState } from "react";

export default function App() {
  const [loaded, setLoaded] = useState(false);
  const [showArticles, setShowArticles] = useState(false);

  useEffect(() => {
    setTimeout(() => setLoaded(true), 100);
  }, []);

  return (
    <main
      className={`min-h-screen bg-gradient-to-br from-[#0a0026] via-[#001f4d] to-[#001322] text-white font-sans overflow-hidden relative transition-opacity duration-1000 ${
        loaded ? "opacity-100" : "opacity-0"
      }`}
    >
      {/* Navbar */}
      <nav className="sticky top-0 z-30 bg-[#0f0c29]/80 backdrop-blur-md border-b border-green-400 shadow-lg flex items-center justify-between p-6">
        <Link to="/">
          <img
            src="/logo.png"
            alt="Lensless Logo"
            className="w-12 h-12 object-contain hover:scale-110 transition-transform duration-300"
          />
        </Link>
        <ul className="flex gap-6 text-sm text-gray-300">
          <li>
            <Link to="/" className="hover:text-green-300 transition">
              Home
            </Link>
          </li>
          <li>
            <a
              href="https://github.com/LCAV/LenslessPiCam"
              target="_blank"
              className="hover:text-green-300 transition"
              rel="noreferrer"
            >
              GitHub
            </a>
          </li>
          <li>
            <a
              href="mailto:info@lenslesspicam.com"
              className="hover:text-green-300 transition"
            >
              Contact
            </a>
          </li>
        </ul>
      </nav>

      {/* Hero Section */}
      <section className="relative z-10 flex flex-col md:flex-row items-center justify-between p-10 md:p-20">
        <div className="w-full md:w-1/2 mb-10 md:mb-0 flex justify-center">
          <img
            src="/logo.png"
            alt="Lensless PiCam Logo"
            className="w-48 h-48 object-contain drop-shadow-xl animate-float"
          />
        </div>
        <div className="w-full md:w-1/2 text-center md:text-left">
          <h1 className="text-5xl md:text-6xl font-extrabold mb-4 bg-gradient-to-r from-green-400 to-teal-300 text-transparent bg-clip-text">
            Lensless PiCam Kit
          </h1>
          <p className="text-lg text-green-300 mb-6">
            All-in-One Starter Kit for Computational Imaging
          </p>
          <Link
            to="/demo"
            className="bg-green-400 text-black hover:bg-green-300 text-lg px-6 py-3 rounded-full inline-block transform transition hover:scale-110 shadow-lg"
          >
            Try our Online Demo
          </Link>
        </div>
      </section>

      {/* Why Lensless Section */}
      <section className="p-10 md:p-20 text-center bg-[#101028]/80">
        <h2 className="text-3xl font-bold mb-10 text-green-300">
          Why Lensless?
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-6xl mx-auto">
          <div className="bg-[#1b1a3d] rounded-xl p-6 border border-green-500 shadow-md hover:scale-105 transition">
            <h3 className="text-lg font-semibold text-white mb-2">
              Reduced size, weight & cost
            </h3>
            <p className="text-sm text-gray-300">
              Lensless designs eliminate bulky optical components, making systems lighter, smaller and cheaper.
            </p>
          </div>
          <div className="bg-[#1b1a3d] rounded-xl p-6 border border-green-500 shadow-md hover:scale-105 transition">
            <h3 className="text-lg font-semibold text-white mb-2">
              Visual privacy
            </h3>
            <p className="text-sm text-gray-300">
              Raw images captured are unintelligible to humans, offering native privacy by design.
            </p>
          </div>
          <div className="bg-[#1b1a3d] rounded-xl p-6 border border-green-500 shadow-md hover:scale-105 transition">
            <h3 className="text-lg font-semibold text-white mb-2">
              Compressive imaging
            </h3>
            <p className="text-sm text-gray-300">
              Captures and reconstructs only what’s necessary, reducing data and enabling smarter sensing.
            </p>
          </div>
        </div>
      </section>

      {/* About Section */}
      <section className="bg-[#12103a]/80 p-10 md:p-20 text-center relative z-10">
        <h2 className="text-3xl font-bold mb-6 text-green-300">About the Project</h2>
        <p className="text-lg max-w-4xl mx-auto text-gray-300">
          The Lensless PiCam Kit is an affordable, modular development platform for lensless imaging.
          It includes 3D-printable hardware, open-source software, and tools for integrating machine learning into the imaging pipeline.
        </p>
      </section>

      {/* Documentation Section */}
      <section className="bg-[#14132f] p-10 md:p-20">
        <h2 className="text-3xl font-bold mb-10 text-center">Documentation</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-6xl mx-auto">
          {/* Medium Articles Section */}
          <div className="bg-[#222150] rounded-lg p-6 border border-green-400">
            <h3 className="text-xl font-semibold mb-2 text-white">Medium Articles</h3>
            <p className="text-sm mb-4 text-gray-300">
              Explore our insights and development stories on Medium.
            </p>
            <button
              onClick={() => setShowArticles(!showArticles)}
              className="text-green-300 underline text-sm hover:text-green-200 transition"
            >
              {showArticles ? "Hide Articles" : "View Articles"}
            </button>
            {showArticles && (
              <ul className="list-disc list-inside mt-4 text-sm text-green-300 space-y-2">
                <li>
                  <a
                    href="https://medium.com/@imane/affordable-lensless-kit-overview"
                    target="_blank"
                    rel="noreferrer"
                    className="underline"
                  >
                    1. Affordable Lensless Kit – Overview
                  </a>
                </li>
                <li>
                  <a
                    href="https://medium.com/@imane/lensless-picam-hardware-design"
                    target="_blank"
                    rel="noreferrer"
                    className="underline"
                  >
                    2. Hardware Design Insights
                  </a>
                </li>
                <li>
                  <a
                    href="https://medium.com/@imane/ml-for-lensless-reconstruction"
                    target="_blank"
                    rel="noreferrer"
                    className="underline"
                  >
                    3. ML for Lensless Reconstruction
                  </a>
                </li>
              </ul>
            )}
          </div>

          {/* Other Documentation Cards */}
          <div className="bg-[#222150] rounded-lg p-6 border border-green-400">
            <h3 className="text-xl font-semibold mb-2">Software & Firmware</h3>
            <p className="text-sm mb-4">Arduino and Python code examples hosted on GitHub.</p>
            <a href="https://github.com/LCAV/LenslessPiCam" target="_blank" rel="noreferrer" className="text-green-300 underline">Browse GitHub</a>
          </div>
          <div className="bg-[#222150] rounded-lg p-6 border border-green-400">
            <h3 className="text-xl font-semibold mb-2">Full Online Documentation</h3>
            <p className="text-sm mb-4">Setup, imaging, firmware flashing, ML integration, and deployment.</p>
            <a href="https://lensless.readthedocs.io/en/latest/index.html" target="_blank" rel="noreferrer" className="text-green-300 underline">View Full Docs</a>
          </div>
        </div>
      </section>

      {/* Product Section */}
      <section className="bg-[#0f0c29]/80 p-10 md:p-20 text-center">
        <h2 className="text-3xl font-bold mb-6 text-green-300">Our Product</h2>
        <p className="mb-4 text-lg">
          Prototype kit for lensless imaging. Fixed price:{" "}
          <span className="font-bold text-white">$100</span>
        </p>
        <Link
          to="/product"
          className="bg-green-400 text-black hover:bg-green-300 text-lg px-6 py-3 rounded-full inline-block transform transition hover:scale-110 shadow-xl"
        >
          Discover Our Product
        </Link>
      </section>

      {/* Footer */}
      <footer className="bg-[#0d0b26] text-center p-6 text-sm text-gray-400 relative z-10">
        <p>
          Contact:{" "}
          <a
            href="mailto:info@lenslesspicam.com"
            className="underline text-green-300"
          >
            info@lenslesspicam.com
          </a>
        </p>
        <p>© 2025 Lensless PiCam | Created by Imane Raihane at EPFL</p>
      </footer>

      {/* Floating Animation */}
      <style>
        {`
        @keyframes float {
          0% { transform: translateY(0); }
          50% { transform: translateY(-6px); }
          100% { transform: translateY(0); }
        }
        .animate-float {
          animation: float 6s ease-in-out infinite;
        }
        `}
      </style>
    </main>
  );
}
