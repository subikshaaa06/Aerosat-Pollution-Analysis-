import './App.css';
import bgVideo from './assets/background.mp4';

function App() {
  const areas = [
    { label: 'Rohini/Pitampura Area', key: 'rohini_pitampura_area', site: '01' },
    { label: 'Dwarka Sector Area', key: 'dwarka_sector_area', site: '02' },
    { label: 'Lodhi Road Area', key: 'lodhi_road_area', site: '03' },
    { label: 'Narela', key: 'narela', site: '04' },
    { label: 'Okhla', key: 'okhla', site: '05' },
    { label: 'Bawana', key: 'bawana', site: '06' },
    { label: 'Wazirpur', key: 'wazirpur', site: '07' }
  ];

  return (
    <div className="app">
      <video className="background-video" autoPlay loop muted>
        <source src={bgVideo} type="video/mp4" />
      </video>

      <div className="landing-shell">
        <div className="landing-copy">
          <div className="live-badge">
            <span className="live-dot"></span>
            LIVE FORECAST NETWORK
          </div>
          <h1>AeroSat Delhi</h1>
          <p>Short-term ground-level O3 and NO2 forecasting from satellite and reanalysis signals.</p>
        </div>

        <div className="glass-box">
          <div className="panel-header-line"></div>
          <h2>Choose Area</h2>

          <div className="station-grid">
            {areas.map((area) => (
              <button
                className="station-chip"
                key={area.key}
                type="button"
                onClick={() => {
                  window.location.href = `dashboard.html?location=${area.key}`;
                }}
              >
                <span>SITE {area.site}</span>
                {area.label}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
