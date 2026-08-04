import { useState, useEffect, useRef } from "react";

const API_URL = import.meta.env.VITE_API_URL || "https://fyp-inx5.onrender.com";

export default function App() {
  const [sketch, setSketch] = useState(null);
  const [preview, setPreview] = useState(null);
  const [description, setDescription] = useState("A fun game with custom mechanics");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [showDownloadModal, setShowDownloadModal] = useState(false);

  const [downloadInfo, setDownloadInfo] = useState({
    downloaded_gb: 0,
    target_gb: 6.5,
    percent: 0,
    speed_mbps: 0,
    file_name: "Checking models...",
    api_key_configured: false,
    api_key_preview: "Checking...",
    is_cached: false,
    is_downloading: false,
  });

  const [status, setStatus] = useState({
    step: "Starting...",
    progress: 0,
    details: "",
  });
  const [elapsed, setElapsed] = useState(0);

  const timerRef = useRef(null);
  const pollRef = useRef(null);
  const downloadPollRef = useRef(null);

  // Poll download status on load and periodically
  useEffect(() => {
    fetchDownloadStatus();
    downloadPollRef.current = setInterval(fetchDownloadStatus, 800);
    return () => clearInterval(downloadPollRef.current);
  }, []);

  async function fetchDownloadStatus() {
    try {
      const res = await fetch(`${API_URL}/download-status`);
      if (res.ok) {
        const data = await res.json();
        setDownloadInfo(data);
      }
    } catch (err) {
      // ignore glitches
    }
  }

  async function handleStartDownload() {
    try {
      await fetch(`${API_URL}/start-download`, { method: "POST" });
      fetchDownloadStatus();
    } catch (err) {
      console.error(err);
    }
  }

  function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  }

  function handleFile(e) {
    const file = e.target.files[0];
    if (!file) return;
    setSketch(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  }

  async function handleGenerate() {
    if (!sketch) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setElapsed(0);

    const jobId = Math.random().toString(36).substring(2, 10);
    setStatus({
      step: "Uploading sketch file...",
      progress: 5,
      details: "Connecting to FastAPI backend",
    });

    // Start timer counter
    timerRef.current = setInterval(() => {
      setElapsed((prev) => prev + 1);
    }, 1000);

    // Start real-time status polling
    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch(`${API_URL}/status/${jobId}`);
        if (res.ok) {
          const data = await res.json();
          setStatus(data);
          if (data.status === "completed" && data.result) {
            setResult(data.result);
            setLoading(false);
            clearInterval(timerRef.current);
            clearInterval(pollRef.current);
          } else if (data.status === "error") {
            setError(data.error || "Generation failed");
            setLoading(false);
            clearInterval(timerRef.current);
            clearInterval(pollRef.current);
          }
        }
      } catch (err) {
        // ignore polling glitches
      }
    }, 1000);

    const form = new FormData();
    form.append("sketch", sketch);
    form.append("description", description);
    form.append("job_id", jobId);

    try {
      const res = await fetch(`${API_URL}/generate`, {
        method: "POST",
        body: form,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Generation request failed");
    } catch (err) {
      setError(err.message);
      setLoading(false);
      clearInterval(timerRef.current);
      clearInterval(pollRef.current);
    }
  }

  const isRacingGenre = result?.game_plan?.genre?.toLowerCase().includes("rac") || result?.game_plan?.genre?.toLowerCase().includes("car");

  return (
    <div className="app">
      {/* Top Status Banner */}
      <div className="status-bar">
        <div className="status-pill">
          <span>🗝️ OpenAI API Key: </span>
          {downloadInfo.api_key_configured ? (
            <span className="badge-success">Configured ✅ ({downloadInfo.api_key_preview})</span>
          ) : (
            <span className="badge-danger">Not Entered ❌</span>
          )}
        </div>

        <button className="download-modal-btn" onClick={() => setShowDownloadModal(true)}>
          📥 Model Download Progress ({downloadInfo.downloaded_gb}GB / {downloadInfo.target_gb}GB)
        </button>
      </div>

      <header>
        <h1>Sketch → Game Asset Generator</h1>
        <p className="subtitle">
          Florence-2 Vision Analysis • GPT-4o Open-Ended Game Planning • 16-Bit Organic Sprites • CLIP Selection
        </p>
      </header>

      <div className="panel">
        <label className="upload-box">
          {preview ? (
            <img src={preview} alt="sketch preview" />
          ) : (
            <div className="upload-placeholder">
              <span className="upload-icon">🎨</span>
              <span>Click to upload your hand-drawn level sketch</span>
            </div>
          )}
          <input type="file" accept="image/*" onChange={handleFile} hidden />
        </label>

        <div className="input-group">
          <label>Game Theme & Mechanic Description</label>
          <input
            className="desc-input"
            type="text"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="e.g. 'a medieval castle with knight and dragon' or 'a retro racing highway' or 'top-down shooter'"
          />
        </div>

        <button
          className="generate-btn"
          onClick={handleGenerate}
          disabled={!sketch || loading}
        >
          {loading ? (
            <span className="btn-loading">
              <span className="spinner"></span> Generating Level...
            </span>
          ) : (
            "🚀 Generate Level Assets"
          )}
        </button>

        {error && <div className="error">❌ {error}</div>}
      </div>

      {loading && (
        <div className="progress-card">
          <div className="progress-header">
            <div className="progress-title">
              <span className="pulse-dot"></span>
              <strong>{status.step || "Processing..."}</strong>
            </div>
            <div className="timer">⏱️ {formatTime(elapsed)}</div>
          </div>

          <div className="progress-bar-bg">
            <div
              className="progress-bar-fill"
              style={{ width: `${Math.max(status.progress || 0, 5)}%` }}
            ></div>
          </div>

          <div className="progress-footer">
            <span className="details-text">
              {status.details || "Working on AI pipeline..."}
            </span>
            <span className="percent-text">{status.progress || 5}%</span>
          </div>

          <div className="steps-checklist">
            <div className={`step-item ${status.progress >= 10 ? "active" : ""}`}>
              <span>1. Florence Vision</span>
            </div>
            <div className={`step-item ${status.progress >= 25 ? "active" : ""}`}>
              <span>2. GPT-4o Plan</span>
            </div>
            <div className={`step-item ${status.progress >= 40 ? "active" : ""}`}>
              <span>3. SDXL/Procedural Sprites</span>
            </div>
            <div className={`step-item ${status.progress >= 88 ? "active" : ""}`}>
              <span>4. CLIP Scoring</span>
            </div>
            <div className={`step-item ${status.progress >= 95 ? "active" : ""}`}>
              <span>5. Playability BFS</span>
            </div>
          </div>
        </div>
      )}

      {/* Model Download & System Status Modal */}
      {showDownloadModal && (
        <div className="modal-overlay" onClick={() => setShowDownloadModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>📥 Real-Time System & Model Download Tracker</h2>
              <button className="close-btn" onClick={() => setShowDownloadModal(false)}>✖</button>
            </div>

            <div className="modal-body">
              <div className="metric-box">
                <div className="metric-title">🗝️ OpenAI API Key Status</div>
                <div className="metric-value">
                  {downloadInfo.api_key_configured ? (
                    <span className="text-success">Entered & Active ✅ ({downloadInfo.api_key_preview})</span>
                  ) : (
                    <span className="text-danger">Missing API Key ❌</span>
                  )}
                </div>
              </div>

              <div className="metric-grid">
                <div className="metric-box">
                  <div className="metric-title">⚡ Live Download Speed</div>
                  <div className="metric-value">
                    {downloadInfo.speed_mbps > 0 ? (
                      <span className="text-success">{downloadInfo.speed_mbps} MB/s</span>
                    ) : (
                      <span>{downloadInfo.speed_mbps} MB/s</span>
                    )}
                  </div>
                </div>
                <div className="metric-box">
                  <div className="metric-title">📦 Total Cached Size</div>
                  <div className="metric-value">{downloadInfo.downloaded_gb} / {downloadInfo.target_gb} GB ({downloadInfo.percent}%)</div>
                </div>
              </div>

              <div className="modal-progress-bg">
                <div
                  className="modal-progress-fill"
                  style={{ width: `${Math.min(downloadInfo.percent, 100)}%` }}
                ></div>
              </div>

              <div className="metric-box">
                <div className="metric-title">📁 Active Downloading File</div>
                <div className="metric-file">{downloadInfo.file_name}</div>
              </div>

              <div className="action-row">
                {!downloadInfo.is_cached && (
                  <button className="start-download-btn" onClick={handleStartDownload}>
                    ▶️ Start / Resume Background Download Now
                  </button>
                )}
              </div>

              <div className="status-note">
                {downloadInfo.is_cached ? (
                  <span className="text-success">✅ All AI models are fully cached on disk! Ready for instant offline level generation.</span>
                ) : downloadInfo.is_downloading ? (
                  <span className="text-warning">⚡ Actively downloading SDXL base model weights from Hugging Face...</span>
                ) : (
                  <span className="text-warning">ℹ️ Click 'Start / Resume Background Download Now' or 'Generate Level' to start downloading model weights.</span>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {result && (
        <div className="results">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '15px', marginBottom: '15px' }}>
            <h2 style={{ margin: 0 }}>🎉 AAA Level Generation Complete!</h2>
            <a
              href={`${API_URL}${result.zip_url || `/download-assets/${result.job_id}`}`}
              download
              className="download-all-btn"
              style={{
                background: 'linear-gradient(135deg, #10b981, #059669)',
                color: '#ffffff',
                padding: '10px 20px',
                borderRadius: '8px',
                fontWeight: 'bold',
                textDecoration: 'none',
                boxShadow: '0 4px 12px rgba(16, 185, 129, 0.3)',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '8px',
                fontSize: '0.95rem'
              }}
            >
              📦 Download All Game Assets (.ZIP)
            </a>
          </div>
          
          <div className="game-meta-row" style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', marginBottom: '15px' }}>
            <span className="badge-genre" style={{ background: '#2563eb', color: '#fff', padding: '6px 14px', borderRadius: '20px', fontWeight: 'bold' }}>
              🎮 Genre: {result.game_plan?.genre || "Platformer"}
            </span>
            <span className="badge-camera" style={{ background: '#7c3aed', color: '#fff', padding: '6px 14px', borderRadius: '20px', fontWeight: 'bold' }}>
              📹 Camera: {result.game_plan?.camera?.style || "Side Camera"}
            </span>
            <span className="badge-status" style={{ background: '#059669', color: '#fff', padding: '6px 14px', borderRadius: '20px', fontWeight: 'bold' }}>
              ✅ Status: {result.playability?.status || "PLAYABLE"}
            </span>
          </div>

          {result.game_plan?.sketch_interpretation && (
            <div
              className="sketch-interpretation-card"
              style={{
                background: '#1e293b',
                borderLeft: '4px solid #2563eb',
                borderRadius: '8px',
                padding: '14px 18px',
                marginBottom: '20px',
                textAlign: 'left'
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 'bold', color: '#93c5fd', marginBottom: '6px' }}>
                <span>🎨</span>
                <span>What the AI understood from your sketch:</span>
              </div>
              <blockquote style={{ margin: 0, color: '#e2e8f0', fontSize: '0.98rem', fontStyle: 'italic', lineHeight: '1.5' }}>
                "{result.game_plan.sketch_interpretation}"
              </blockquote>
            </div>
          )}

          <div className="asset-grid">
            <div className="result-card">
              <h3>Composed Level Scene</h3>
              <img src={`${API_URL}${result.urls.scene}`} alt="generated scene" />
              <a
                href={`${API_URL}${result.urls.scene}`}
                download="scene.png"
                style={{ display: 'block', marginTop: '10px', color: '#3b82f6', textDecoration: 'underline', fontSize: '0.85rem' }}
              >
                📥 Download Scene Image
              </a>
            </div>
            <div className="result-card">
              <h3>Animated Level Preview (H.264 Video)</h3>
              <video src={`${API_URL}${result.urls.preview}`} controls autoPlay loop />
              <a
                href={`${API_URL}${result.urls.preview}`}
                download="preview.mp4"
                style={{ display: 'block', marginTop: '10px', color: '#3b82f6', textDecoration: 'underline', fontSize: '0.85rem' }}
              >
                📥 Download Preview Video (.MP4)
              </a>
            </div>
          </div>

          {/* Individual Sprites Gallery */}
          <div className="panel" style={{ marginTop: '20px', textAlign: 'left' }}>
            <h3>🖼️ Individual Sprite & Asset Package</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '15px', marginTop: '15px' }}>
              {['player', 'enemy', 'platform_tile', 'background'].map((assetName) => {
                let displayName = assetName.replace('_', ' ');
                if (assetName === 'enemy' && isRacingGenre) {
                  displayName = 'Rival Car';
                } else if (assetName === 'player' && isRacingGenre) {
                  displayName = 'Player Car';
                }
                return (
                  <div key={assetName} style={{ background: '#0f172a', padding: '12px', borderRadius: '8px', border: '1px solid #334155', textAlign: 'center' }}>
                    <h4 style={{ margin: '0 0 10px 0', textTransform: 'capitalize', fontSize: '0.9rem', color: '#93c5fd' }}>
                      {displayName}
                    </h4>
                    <img
                      src={`${API_URL}/files/${result.job_id}/best_${assetName}.png?v=${result.job_id}`}
                      alt={displayName}
                      style={{ width: '100%', height: '110px', objectFit: 'contain', background: '#1e293b', borderRadius: '6px', padding: '4px' }}
                    />
                    <a
                      href={`${API_URL}/files/${result.job_id}/best_${assetName}.png`}
                      download={`best_${assetName}.png`}
                      style={{
                        display: 'inline-block',
                        marginTop: '10px',
                        background: '#3b82f6',
                        color: 'white',
                        padding: '5px 10px',
                        borderRadius: '5px',
                        fontSize: '0.78rem',
                        textDecoration: 'none',
                        fontWeight: 'bold'
                      }}
                    >
                      📥 Download Sprite
                    </a>
                  </div>
                );
              })}
            </div>
          </div>

          {result.game_plan?.gameplay_systems && (
            <div className="panel" style={{ marginTop: '20px', textAlign: 'left' }}>
              <h3>⚡ Gameplay Systems & Physics Specification</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '15px', marginTop: '10px' }}>
                <div><strong>❤️ Health:</strong> {result.game_plan.gameplay_systems.health || 100} HP</div>
                <div><strong>🎮 Lives:</strong> {result.game_plan.gameplay_systems.lives || 3}</div>
                <div><strong>🏆 Win Condition:</strong> {result.game_plan.gameplay_systems.win_condition}</div>
                <div><strong>⚙️ Gravity:</strong> {result.game_plan.physics_data?.gravity || 9.8} m/s²</div>
                <div><strong>🏃 Speed:</strong> {result.game_plan.physics_data?.move_speed || 5.0} m/s</div>
                <div><strong>🦘 Jump Force:</strong> {result.game_plan.physics_data?.jump_force || 12.5}</div>
              </div>
            </div>
          )}

          <details>
            <summary>View AAA Game Plan & System Architecture JSON</summary>
            <pre>{JSON.stringify(result.game_plan, null, 2)}</pre>
          </details>
          <details>
            <summary>View Grid Layout JSON</summary>
            <pre>{JSON.stringify(result.layout, null, 2)}</pre>
          </details>
        </div>
      )}
    </div>
  );
}
