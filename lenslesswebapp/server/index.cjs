const express = require('express');
const cors = require('cors');
const { exec } = require('child_process');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = 5000;

app.use(cors());
app.use(express.json());

// Helper function to execute a command and return a Promise
function runCommand(command) {
  console.log(`Running command: ${command}`);
  return new Promise((resolve, reject) => {
    exec(command, (error, stdout, stderr) => {
      if (error) {
        console.error(`Error running command: ${command}`);
        console.error(stderr);
        reject(stderr);
      } else {
        console.log(`Command output: ${stdout}`);
        resolve(stdout);
      }
    });
  });
}

app.post('/run-demo', async (req, res) => {
  try {
    console.log("[SERVER] /run-demo endpoint hit");

    // 1. Capture
    console.log("[SERVER] Starting on-device capture");
    await runCommand(`PYTHONPATH=.. python3 ../scripts/measure/on_device_capture.py \
      sensor=rpi_hq bayer=True fn=test_psf/raw_data.png \
      exp=1 iso=100 config_pause=2 sensor_mode=0 \
      nbits_out=12 legacy=True rgb=False gray=False sixteen=True -- plot=True`);


    // 2. Color correction
    console.log("[SERVER] Running color correction");
    await runCommand(`python3 ../scripts/measure/analyze_image.py \
      --fp test_psf/raw_data.png \
      --bayer --gamma 2.2 --rg 2.0 --bg 1.1 --save test_psf/psf_rgb.png`);

    // 3. Autocorrelation (assumes psf_1mm/raw_data.png already exists)
    console.log("[SERVER] Running autocorrelation analysis");
    await runCommand(`python3 ../scripts/measure/analyze_image.py \
      --fp psf_1mm/raw_data.png \
      --bayer --gamma 2.2 --rg 2.0 --bg 1.1 --lensless`);

    // Read both images as base64
    console.log("[SERVER] Reading images and sending response");
    const psfBuffer = fs.readFileSync(path.join(__dirname, '../test_psf/psf_rgb.png'));
    const autocorrBuffer = fs.readFileSync(path.join(__dirname, '../psf_1mm/autocorr.png'));

    res.json({
      psf: psfBuffer.toString('base64'),
      autocorr: autocorrBuffer.toString('base64')
    });

  } catch (err) {
    console.error("[SERVER] Error in /run-demo", err);
    res.status(500).send({ error: 'Demo failed to run', details: err });
  }
});

app.listen(PORT, () => {
  console.log(`Demo backend running on http://localhost:${PORT}`);
});
