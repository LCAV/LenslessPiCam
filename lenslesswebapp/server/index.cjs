const express = require('express');
const cors = require('cors');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const https = require('https');

const app = express();
const PORT = 5000;

app.use(cors());
app.use(express.json());

// Helper function to execute a shell command with real-time output
function runCommandLive(command, args, cwd) {
  console.log(`\n[COMMAND] ${command} ${args.join(' ')}\n`);
  return new Promise((resolve, reject) => {
    const proc = spawn(command, args, { cwd, shell: true });

    proc.stdout.on('data', (data) => {
      process.stdout.write(`[stdout] ${data}`);
    });

    proc.stderr.on('data', (data) => {
      process.stderr.write(`[stderr] ${data}`);
    });

    proc.on('close', (code) => {
      if (code === 0) {
        console.log(` Command completed: ${command}`);
        resolve();
      } else {
        reject(new Error(`Command failed with exit code ${code}`));
      }
    });
  });
}

app.post('/run-demo', async (req, res) => {
  try {
    console.log("[SERVER] /run-demo endpoint hit");

    const repoRoot = path.join(__dirname, '..');

    // 1. Create test_psf directory
    await runCommandLive('mkdir', ['-p', 'test_psf'], repoRoot);

    // 2. Run on-device capture
    await runCommandLive(
      path.join(repoRoot, 'lensless_env/bin/python'),
      [
        'scripts/measure/on_device_capture.py',
        'sensor=rpi_hq',
        'bayer=True',
        'fn=test_psf/raw_data',
        'exp=1',
        'iso=100',
        'config_pause=2',
        'sensor_mode=0',
        'nbits_out=12',
        'legacy=True',
        'rgb=False',
        'gray=False',
        'sixteen=True',
        'down=4'
      ],
      repoRoot
    );

    // 3. Run color correction
    await runCommandLive(
      path.join(repoRoot, 'lensless_env/bin/python'),
      [
        'scripts/measure/analyze_image.py',
        '--fp', 'test_psf/raw_data.png',
        '--bayer',
        '--gamma', '2.2',
        '--rg', '2.0',
        '--bg', '1.1',
        '--save', 'test_psf/psf_rgb.png'
      ],
      repoRoot
    );

    // 4. Run autocorrelation
    await runCommandLive(
      path.join(repoRoot, 'lensless_env/bin/python'),
      [
        'scripts/measure/analyze_image.py',
        '--fp', 'test_psf/raw_data.png',
        '--bayer',
        '--gamma', '2.2',
        '--rg', '2.0',
        '--bg', '1.1',
        '--lensless'
      ],
      repoRoot
    );

    // 5. Read images and respond
    const psfBuffer = fs.readFileSync(path.join(repoRoot, 'test_psf/psf_rgb.png'));
    const autocorrBuffer = fs.readFileSync(path.join(repoRoot, 'psf_1mm/autocorr.png'));

    res.json({
      psf: psfBuffer.toString('base64'),
      autocorr: autocorrBuffer.toString('base64')
    });

  } catch (err) {
    console.error("[SERVER] Error in /run-demo", err);
    res.status(500).send({ error: 'Demo failed to run', details: err.toString() });
  }
});

// HTTPS server
const key = fs.readFileSync(path.join(__dirname, '../certs/key.pem'));
const cert = fs.readFileSync(path.join(__dirname, '../certs/cert.pem'));

https.createServer({ key, cert }, app).listen(PORT, '0.0.0.0', () => {
  console.log(`Demo backend running on https://128.179.187.191:${PORT}`);
});
