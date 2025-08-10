document.addEventListener('DOMContentLoaded', () => {
    // --- UI Elements ---
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const detectionCanvas = document.getElementById('detectionCanvas');
    const trajectoryCanvas = document.getElementById('trajectoryCanvas');
    const startCameraBtn = document.getElementById('startCamera');
    const stopCameraBtn = document.getElementById('stopCamera');
    const captureBtn = document.getElementById('capture');
    const detectBtn = document.getElementById('detect');
    const calculateTrajectoryBtn = document.getElementById('calculateTrajectory');
    const launchMissileBtn = document.getElementById('launchMissile');
    const fileInput = document.getElementById('fileInput');
    const launchAngleSlider = document.getElementById('launchAngle');
    const launchAngleValue = document.getElementById('angleValue');
    const launchDirectionSlider = document.getElementById('launchDirection');
    const launchDirectionValue = document.getElementById('directionValue');
    const detectionResults = document.getElementById('detectionResults');
    const weaponDetectionResults = document.getElementById('weaponDetectionResults');
    const scanningOverlay = document.getElementById('scanningOverlay');
    const systemLogs = document.getElementById('systemLogs');
    const targetCoordinates = document.getElementById('targetCoordinates');
    const targetDistance = document.getElementById('targetDistance');
    const debugConsole = document.getElementById('debug-console');
    const modelStatus = document.getElementById('modelStatus');
    const cameraStatus = document.getElementById('cameraStatus');
    const lastScan = document.getElementById('lastScan');
    const objectCount = document.getElementById('objectCount');
    const avgConfidence = document.getElementById('avgConfidence');
    const statusDot = document.querySelector('.status-dot');
    const statusText = document.querySelector('.status-text');
    const networkAssetsTbody = document.getElementById('network-assets');
    const downlinkBar = document.getElementById('downlink-bar');
    const uplinkBar = document.getElementById('uplink-bar');
    const downlinkValue = document.getElementById('downlink-value');
    const uplinkValue = document.getElementById('uplink-value');
    const diagnosticsLog = document.getElementById('diagnostics-log');
    const runDiagnosticsBtn = document.getElementById('run-diagnostics-btn');
    const cpuUsage = document.getElementById('cpu-usage');
    const gpuUsage = document.getElementById('gpu-usage');
    const memUsage = document.getElementById('mem-usage');
    const netLatency = document.getElementById('net-latency');
    const purgeCacheBtn = document.getElementById('purge-cache-btn');
    const recalibrateSensorsBtn = document.getElementById('recalibrate-sensors-btn');
    const rebootSystemBtn = document.getElementById('reboot-system-btn');
    const scopeEoBtn = document.getElementById('scope-eo');
    const scopeIrBtn = document.getElementById('scope-ir');

    // --- State Variables ---
    let stream = null;
    let isScanning = false;
    let currentImage = null;
    let lastPredictions = [];
    let selectedTarget = null;
    let networkInterval = null;
    let performanceInterval = null;
    let currentScope = 'EO';

    // --- Core Functions ---
    function updateClock() {
        const now = new Date();
        document.getElementById("currentTime").textContent = now.toLocaleTimeString('en-US', { hour12: false });
        document.getElementById("currentDate").textContent = now.toLocaleDateString('en-US');
    }

    function addSystemLog(message) {
        const logEntry = document.createElement('div');
        const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false });
        logEntry.textContent = `[${timestamp}] ${message}`;
        systemLogs.insertBefore(logEntry, systemLogs.firstChild);
    }

    function updateStatus(type, message) {
        if (type === 'system') statusText.textContent = message;
        else if (type === 'model') modelStatus.textContent = message;
        else if (type === 'camera') cameraStatus.textContent = message;
        else if (type === 'scan') lastScan.textContent = message;
    }

    function switchToTab(tabName) {
        document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));
        document.querySelectorAll('.nav-section').forEach(section => section.classList.remove('active'));
        const activeItem = Array.from(document.querySelectorAll('.nav-item')).find(el => el.querySelector('.nav-text').textContent.toLowerCase() === tabName);
        if (activeItem) activeItem.classList.add('active');
        document.getElementById(`section-${tabName}`).classList.add('active');

        if (tabName === 'network') startNetworkSimulation();
        else stopNetworkSimulation();

        if (tabName === 'system') startPerformanceMonitor();
        else stopPerformanceMonitor();
    }
    
    // --- System & Network Simulations ---
    function startPerformanceMonitor() {
        if (performanceInterval) return;
        performanceInterval = setInterval(() => {
            cpuUsage.textContent = `${(Math.random() * 30 + 10).toFixed(1)}%`;
            gpuUsage.textContent = `${(Math.random() * 50 + 25).toFixed(1)}%`;
            memUsage.textContent = `${(Math.random() * 1024 + 2048).toFixed(0)} MB`;
            netLatency.textContent = `${(Math.random() * 20 + 30).toFixed(0)} ms`;
        }, 1500);
    }

    function stopPerformanceMonitor() {
        clearInterval(performanceInterval);
        performanceInterval = null;
    }

    function runDiagnostics() {
        diagnosticsLog.innerHTML = '';
        runDiagnosticsBtn.disabled = true;
        const checks = [
            { name: 'AI CORE (YOLOv8x)', status: 'OK' },
            { name: 'DEPTH SENSOR (MiDaS)', status: 'OK' },
            { name: 'API ENDPOINT', status: 'OK' },
            { name: 'CACHE INTEGRITY', status: 'WARN' },
        ];
        let delay = 0;
        checks.forEach(check => {
            setTimeout(() => {
                const log = document.createElement('div');
                log.innerHTML = `CHECKING ${check.name}... <span class="log-${check.status.toLowerCase()}">${check.status}</span>`;
                diagnosticsLog.appendChild(log);
            }, delay);
            delay += 500;
        });
        setTimeout(() => { runDiagnosticsBtn.disabled = false; }, delay);
    }
    
    function startNetworkSimulation() {
        if (networkInterval) return;
        const mockAssets = [
            { id: 'SAT-25-381-01', location: 'PALMDALE, CA', activity: 'UPLINKING TELEMETRY', status: 'ACTIVE' },
            { id: 'DRN-19-141-11', location: 'AREA 51, NV', activity: 'IDLE', status: 'ACTIVE' },
            { id: 'GND-41-121-50', location: 'PINE GAP, AU', activity: 'MAINTENANCE', status: 'INACTIVE' },
        ];
        networkAssetsTbody.innerHTML = mockAssets.map(asset => `
            <tr>
                <td>${asset.id}</td>
                <td>${asset.location}</td>
                <td class="status-${asset.status.toLowerCase()}">${asset.status}</td>
                <td>${asset.activity}</td>
                <td class="traffic-cell">0.00</td>
                <td class="signal-cell">0%</td>
            </tr>
        `).join('');
        networkInterval = setInterval(() => {
            let totalDownlink = 0, totalUplink = 0;
            networkAssetsTbody.querySelectorAll('tr').forEach(row => {
                if (row.children[2].textContent === 'ACTIVE') {
                    const traffic = Math.random() * 25;
                    row.querySelector('.traffic-cell').textContent = traffic.toFixed(2);
                    row.querySelector('.signal-cell').textContent = `${(70 + Math.random() * 30).toFixed(0)}%`;
                    if (row.children[3].textContent.includes('UPLINK')) totalUplink += traffic;
                    else totalDownlink += traffic;
                }
            });
            downlinkValue.textContent = totalDownlink.toFixed(2);
            uplinkValue.textContent = totalUplink.toFixed(2);
            downlinkBar.style.width = `${(totalDownlink / 50) * 100}%`;
            uplinkBar.style.width = `${(totalUplink / 50) * 100}%`;
        }, 1500);
    }

    function stopNetworkSimulation() {
        clearInterval(networkInterval);
        networkInterval = null;
    }

    // --- Event Listeners ---
    document.querySelectorAll('.nav-item').forEach(item => item.addEventListener('click', () => switchToTab(item.querySelector('.nav-text').textContent.toLowerCase())));
    startCameraBtn.addEventListener('click', async () => {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
            video.srcObject = stream;
            video.play();
            startCameraBtn.disabled = true;
            stopCameraBtn.disabled = false;
            captureBtn.disabled = false;
            updateStatus('camera', 'ACTIVE');
        } catch (err) { addSystemLog('Camera access denied.'); }
    });
    stopCameraBtn.addEventListener('click', () => {
        if (stream) stream.getTracks().forEach(track => track.stop());
        startCameraBtn.disabled = false;
        stopCameraBtn.disabled = true;
        captureBtn.disabled = true;
        detectBtn.disabled = true;
        calculateTrajectoryBtn.disabled = true;
        launchMissileBtn.disabled = true;
        updateStatus('camera', 'STANDBY');
    });
    captureBtn.addEventListener('click', () => {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
        currentImage = canvas.toDataURL('image/jpeg');
        detectionCanvas.width = canvas.width;
        detectionCanvas.height = canvas.height;
        detectionCanvas.getContext('2d').drawImage(canvas, 0, 0);
        detectionCanvas.style.display = 'block';
        video.style.display = 'none';
        detectBtn.disabled = false;
    });
    fileInput.addEventListener('change', (e) => {
        const reader = new FileReader();
        reader.onload = (event) => {
            const img = new Image();
            img.onload = () => {
                detectionCanvas.width = img.width;
                detectionCanvas.height = img.height;
                detectionCanvas.getContext('2d').drawImage(img, 0, 0);
                currentImage = detectionCanvas.toDataURL('image/jpeg');
                detectionCanvas.style.display = 'block';
                video.style.display = 'none';
                detectBtn.disabled = false;
            };
            img.src = event.target.result;
        };
        reader.readAsDataURL(e.target.files[0]);
    });
    detectBtn.addEventListener('click', async () => {
        scanningOverlay.style.display = 'flex';
        const fetchRes = await fetch(currentImage);
        const blob = await fetchRes.blob();
        const formData = new FormData();
        formData.append('file', blob, 'capture.jpg');
        formData.append('scope', currentScope);
        const response = await fetch('http://127.0.0.1:8000/api/analyze', { method: 'POST', body: formData });
        const data = await response.json();
        lastPredictions = data.predictions || [];
        selectedTarget = null;
        drawDetections(detectionCanvas.getContext('2d'), lastPredictions);
        displayDetectionResults(lastPredictions);
        objectCount.textContent = lastPredictions.length;
        const avgConf = lastPredictions.length > 0 ? (lastPredictions.reduce((sum, p) => sum + p.score, 0) / lastPredictions.length) * 100 : 0;
        avgConfidence.textContent = `${avgConf.toFixed(1)}%`;
        updateStatus('scan', new Date().toLocaleTimeString());
        switchToTab('analytics');
        addSystemLog(`Analysis complete. ${lastPredictions.length} targets found.`);
        scanningOverlay.style.display = 'none';
    });
    detectionCanvas.addEventListener('click', (e) => {
        const rect = detectionCanvas.getBoundingClientRect();
        const scaleX = detectionCanvas.width / rect.width;
        const scaleY = detectionCanvas.height / rect.height;
        const x = (e.clientX - rect.left) * scaleX;
        const y = (e.clientY - rect.top) * scaleY;
        const clickedPrediction = lastPredictions.find(p => x >= p.bbox[0] && x <= p.bbox[0] + p.bbox[2] && y >= p.bbox[1] && y <= p.bbox[1] + p.bbox[3]);
        debugConsole.innerHTML = `Click at X:${x.toFixed(0)}, Y:${y.toFixed(0)}\n`;
        if (clickedPrediction) {
            selectedTarget = clickedPrediction;
            debugConsole.innerHTML += `<span class="debug-success">SUCCESS: Target locked: ${selectedTarget.class.toUpperCase()}</span>`;
            const targetX = (selectedTarget.bbox[0] + selectedTarget.bbox[2] / 2).toFixed(0);
            const targetY = (selectedTarget.bbox[1] + selectedTarget.bbox[3] / 2).toFixed(0);
            targetCoordinates.textContent = `X:${targetX}, Y:${targetY}`;
            const distanceInKm = selectedTarget.distance > 0 ? 200 / selectedTarget.distance : 0;
            targetDistance.textContent = `${distanceInKm.toFixed(2)} km`;
            calculateTrajectoryBtn.disabled = false;
            launchMissileBtn.disabled = true;
            switchToTab('targeting');
            drawDetections(detectionCanvas.getContext('2d'), lastPredictions);
        } else {
            debugConsole.innerHTML += `<span class="debug-info">INFO: No target hit.</span>`;
        }
    });
    calculateTrajectoryBtn.addEventListener("click", async () => {
        const distanceInMeters = parseFloat(targetDistance.textContent) * 1000;
        const response = await fetch('http://127.0.0.1:8000/api/trajectory', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ target_distance: distanceInMeters })
        });
        const data = await response.json();
        if (data.path) {
            drawTrajectory(data.path);
            const angle = data.calculated_angle.toFixed(1);
            launchAngleSlider.value = angle;
            launchAngleValue.textContent = `${angle}°`;
            launchMissileBtn.disabled = false;
        } else { addSystemLog(data.error); }
    });
    launchAngleSlider.oninput = (e) => launchAngleValue.textContent = `${e.target.value}°`;
    launchDirectionSlider.oninput = (e) => launchDirectionValue.textContent = `${e.target.value}°`;
    scopeEoBtn.addEventListener('click', () => { currentScope = 'EO'; scopeEoBtn.classList.add('active'); scopeIrBtn.classList.remove('active'); });
    scopeIrBtn.addEventListener('click', () => { currentScope = 'IR'; scopeIrBtn.classList.add('active'); scopeEoBtn.classList.remove('active'); });
    runDiagnosticsBtn.addEventListener('click', runDiagnostics);
    purgeCacheBtn.addEventListener('click', () => addSystemLog('Cache purged successfully.'));
    recalibrateSensorsBtn.addEventListener('click', () => addSystemLog('Sensor recalibration initiated.'));
    rebootSystemBtn.addEventListener('click', () => {
        addSystemLog('SYSTEM REBOOT COMMAND ISSUED.');
        alert('SYSTEM REBOOTING...');
        location.reload();
    });

    // --- Drawing and Display Functions ---
    function drawDetections(ctx, predictions) {
        const img = new Image();
        img.onload = () => {
            ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
            ctx.drawImage(img, 0, 0);
            predictions.forEach(p => {
                const isSelected = selectedTarget && p.bbox.toString() === selectedTarget.bbox.toString();
                ctx.beginPath();
                ctx.rect(p.bbox[0], p.bbox[1], p.bbox[2], p.bbox[3]);
                ctx.lineWidth = isSelected ? 4 : 2;
                ctx.strokeStyle = isSelected ? '#FFD700' : '#00ff9d';
                ctx.stroke();
            });
        };
        img.src = currentImage;
    }

    function displayDetectionResults(predictions) {
        const generalDetections = predictions.filter(p => p.threat_level === 'NONE' || p.threat_level === 'INFO');
        const threatDetections = predictions.filter(p => p.threat_level === 'LOW' || p.threat_level === 'MEDIUM' || p.threat_level === 'HIGH');
        detectionResults.innerHTML = generalDetections.length === 0 ? `<div class="no-results"><p>NO OBJECTS DETECTED</p></div>` : '';
        generalDetections.forEach(p => {
            const item = document.createElement('div');
            item.className = 'detection-item';
            const distance = p.distance > 0 ? `Rel. Distance: ${p.distance.toFixed(2)}` : 'Distance: N/A';
            item.innerHTML = `<h3>${p.class.toUpperCase()}</h3><p>Confidence: ${(p.score * 100).toFixed(1)}%</p><p>${distance}</p>`;
            detectionResults.appendChild(item);
        });
        weaponDetectionResults.innerHTML = threatDetections.length === 0 ? `<div class="no-results"><p>NO THREATS DETECTED</p></div>` : '';
        threatDetections.forEach(p => {
            const item = document.createElement('div');
            item.className = 'detection-item threat';
            const distance = p.distance > 0 ? `Rel. Distance: ${p.distance.toFixed(2)}` : 'Distance: N/A';
            item.innerHTML = `<h3>${p.class.toUpperCase()}<span class="threat-level threat-level-${p.threat_level.toLowerCase()}">${p.threat_level}</span></h3><p>Type: <span class="threat-type">${p.threat_type}</span></p><p>Confidence: ${(p.score * 100).toFixed(1)}%</p><p>${distance}</p>`;
            weaponDetectionResults.appendChild(item);
        });
    }

    function drawTrajectory(path) {
        const tCtx = trajectoryCanvas.getContext('2d');
        tCtx.clearRect(0, 0, trajectoryCanvas.width, trajectoryCanvas.height);
        if (!path || path.length === 0) return;
        const maxHeight = Math.max(...path.map(p => p.y));
        const maxRange = path[path.length - 1].x;
        tCtx.beginPath();
        tCtx.moveTo(10, trajectoryCanvas.height - 10);
        path.forEach(point => {
            const canvasX = 10 + (point.x / maxRange) * (trajectoryCanvas.width - 20);
            const canvasY = (trajectoryCanvas.height - 10) - (point.y / maxHeight) * (trajectoryCanvas.height - 20);
            tCtx.lineTo(canvasX, canvasY);
        });
        tCtx.strokeStyle = "#ff3c3c";
        tCtx.lineWidth = 2;
        tCtx.setLineDash([5, 10]);
        tCtx.stroke();
        tCtx.setLineDash([]);
    }

    // --- Initializations ---
    setInterval(updateClock, 1000);
    updateClock();
    addSystemLog("System initialized.");
});
