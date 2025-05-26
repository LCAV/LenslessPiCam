import { Link } from "react-router-dom";

export default function ProductPage() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-[#0a0026] via-[#001f4d] to-[#001322] text-white font-sans relative overflow-hidden">
      
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

      {/* Header */}
      <header className="text-center mb-12 p-10">
        <h1 className="text-4xl font-extrabold mb-2 bg-gradient-to-r from-green-400 to-teal-300 text-transparent bg-clip-text">
          Lensless PiCam Kit
        </h1>
        <p className="text-green-300 text-lg">Modular, Affordable, and Open-Source</p>
      </header>

      {/* Content */}
      <div className="flex flex-col md:flex-row gap-10 max-w-6xl mx-auto px-6 pb-20">
        {/* Images */}
        <div className="flex-1 grid grid-cols-2 gap-4">
          <div className="bg-[#222150] h-48 rounded-lg flex items-center justify-center text-gray-400 border border-green-400">Main Image</div>
          <div className="bg-[#222150] h-48 rounded-lg flex items-center justify-center text-gray-400 border border-green-400">Thumb 1</div>
          <div className="bg-[#222150] h-48 rounded-lg flex items-center justify-center text-gray-400 border border-green-400">Thumb 2</div>
          <div className="bg-[#222150] h-48 rounded-lg flex items-center justify-center text-gray-400 border border-green-400">Thumb 3</div>
        </div>

        {/* Product Info */}
        <div className="flex-1 bg-[#14132f]/90 p-6 rounded-lg shadow-xl border border-green-500">
          <h2 className="text-2xl font-bold mb-2">$100.00</h2>
          <p className="text-red-400 mb-4">Available Now</p>
          <button className="bg-green-400 text-black font-semibold px-6 py-3 rounded hover:bg-green-300 mb-6 transform transition hover:scale-105">
            Buy Now
          </button>
          <p className="text-sm text-gray-300 leading-relaxed">
            The Lensless PiCam Kit includes all the hardware you need to begin experimenting with computational imaging:
            a custom PCB, image sensor, microcontroller connectors, and full documentation. Compatible with Raspberry Pi and Arduino,
            and designed to support real-time capture and machine learning processing.
          </p>

          <ul className="list-disc list-inside mt-4 text-sm text-gray-200">
            <li>Open-source and educational</li>
            <li>Modular and extensible</li>
            <li>Includes pre-trained models and firmware</li>
            <li>Designed by LCAV – EPFL</li>
          </ul>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-[#0d0b26] text-center p-6 text-sm text-gray-400">
        <p>
          Contact:{" "}
          <a href="mailto:info@lenslesspicam.com" className="underline text-green-300">
            info@lenslesspicam.com
          </a>
        </p>
        <p>© 2025 Lensless PiCam | Created by Imane Raihane at EPFL</p>
      </footer>
    </main>
  );
}
