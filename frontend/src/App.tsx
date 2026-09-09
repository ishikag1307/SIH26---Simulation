import { useState } from 'react'
import ObserverScene from './ObserverScene'
import { useTelemetry } from './useTelemetry'

const degrees = (radians?: number) => (radians === undefined ? '—' : `${(radians * 180 / Math.PI).toFixed(2)}°`)
const value = (number?: number | null, suffix = '') => number === undefined || number === null ? '—' : `${number.toFixed(2)}${suffix}`

export default function App() {
  const { telemetry, connected } = useTelemetry()
  const [activeScenario, setActiveScenario] = useState<string>('uav-ground')

  const switchScenario = async (scenario: string) => {
    setActiveScenario(scenario)
    try {
      await fetch('/api/scenario', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scenario })
      })
    } catch {
      // Ignore if offline
    }
  }

  const exportLog = () => {
    window.location.href = '/api/export_log.csv'
  }

  return (
    <main>
      <header>
        <div>
          <p className="eyebrow">SIH 2026 · TEAM DOOMSDAY SQUAD · ISRO FSOC</p>
          <h1>Virtual Camera Tracking Lab</h1>
        </div>
        <div className={`connection ${connected ? 'online' : ''}`}>
          <span /> {connected ? 'LIVE TELEMETRY' : 'STANDBY (STANDALONE DEMO)'}
        </div>
      </header>

      <section className="workspace">
        <article className="panel observer-panel">
          <div className="panel-heading">
            <div>
              <p className="panel-kicker">OBSERVER VIEW</p>
              <h2>Interactive 3D Relative Motion ({activeScenario.toUpperCase()})</h2>
            </div>
            <p className="hint">Drag to orbit · Scroll to zoom · Right-drag to pan</p>
          </div>
          <div className="scene"><ObserverScene telemetry={telemetry} /></div>
        </article>

        <article className="panel camera-panel">
          <div className="panel-heading">
            <div>
              <p className="panel-kicker">TRACKING CAMERA</p>
              <h2>Synthetic Detector Feed</h2>
            </div>
            <span className={`lock ${telemetry?.locked ? 'locked' : ''}`}>{telemetry?.locked ? 'LOCKED' : 'ACQUIRING'}</span>
          </div>
          <div className="camera-frame">
            <img src="/api/camera.mjpeg" alt="Live synthetic tracking camera feed" onError={(e) => {
              // Fallback placeholder image if server stream is offline
              (e.target as HTMLImageElement).src = 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480"><rect width="100%" height="100%" fill="%23040610"/><text x="50%" y="50%" fill="%234fd8ff" font-family="sans-serif" font-size="16" text-anchor="middle">Open uav_tracking_lab.html for Standalone Mode</text></svg>'
            }} />
          </div>
          <div className="legend">
            <span><i className="opencv" />OpenCV measurement</span>
            <span><i className="kalman" />Kalman estimate</span>
            <span><i className="axis" />Optical axis</span>
          </div>
        </article>
      </section>

      <section className="controls-bar" style={{ margin: '16px 0', display: 'flex', gap: '10px', flexWrap: 'wrap', justifyContent: 'space-between' }}>
        <div className="scenarios" style={{ display: 'flex', gap: '8px' }}>
          {[
            ['uav-ground', 'UAV → Ground'],
            ['uav-uav', 'UAV → UAV'],
            ['sat-ground', 'Sat → Ground'],
            ['sat-sat', 'Sat → Sat']
          ].map(([key, label]) => (
            <button
              key={key}
              onClick={() => switchScenario(key)}
              className={`btn ${activeScenario === key ? 'primary' : 'ghost'}`}
              style={{ padding: '8px 14px', borderRadius: '6px', cursor: 'pointer' }}
            >
              {label}
            </button>
          ))}
        </div>
        <button onClick={exportLog} className="btn ghost" style={{ padding: '8px 16px', cursor: 'pointer' }}>
          Export Performance Log (CSV)
        </button>
      </section>

      <section className="telemetry">
        <article><p>Tracking error</p><strong>{value(telemetry?.tracking_error_px, ' px')}</strong><small>Kalman estimate to boresight</small></article>
        <article><p>Acquisition Time</p><strong>{value(telemetry?.acquisition_time_s, ' s')}</strong><small>Time to initial lock</small></article>
        <article><p>Lock Retention</p><strong>{value(telemetry?.lock_retention_pct, '%')}</strong><small>Frames in lock / total</small></article>
        <article><p>Camera attitude</p><strong>{degrees(telemetry?.camera_pan_rad)} / {degrees(telemetry?.camera_tilt_rad)}</strong><small>Pan / tilt</small></article>
        <article><p>Simulated time</p><strong>{value(telemetry?.simulation_time_s, ' s')}</strong><small>Real-time physics clock</small></article>
      </section>

      <footer>
        <span>Pipeline</span>
        FSOC Scenario Physics <b>→</b> Optical Detector <b>→</b> OpenCV Centroid <b>→</b> 2-Axis Kalman <b>→</b> PID Gimbal Actuator
      </footer>
    </main>
  )
}
