import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import App from './App.tsx'
import ProductPage from './ProductPage.tsx'
import DemoPage from './DemoPage.tsx'
import PhoneCapturePage from './PhoneCapturePage.tsx';

import './index.css' 

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/product" element={<ProductPage />} />
        <Route path="/demo" element={<DemoPage />} />
        <Route path="/phone" element={<PhoneCapturePage />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
)