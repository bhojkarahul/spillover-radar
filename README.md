📡 Spillover Radar

One Health AI Surveillance & Early Outbreak Detection Platform

Built for Hack4Pashudhan | Track 05: One Health Dashboard

Team ID: H4P-VC587GBK

Team: VetTech Trio (Parth Kantharia, Priyal Kantharia, Rahul Bhojkar)

🔗 Live Demo: https://spillover-radar.onrender.com

📖 Overview

Spillover Radar is a geospatial data intelligence dashboard designed to predict and detect zoonotic outbreak patterns by connecting livestock mobility, environmental climate shifts, and human public health indicators. It moves beyond reactive monitoring into predictive forecasting, allowing District Veterinary Officers (DVOs) to deploy resources proactively.

✨ Key Features

🧠 AI-Powered Anomaly Detection: Utilizes Isolation Forest machine learning and CUSUM (Cumulative Sum) models to detect critical risk anomalies and change points in syndromic surveillance data.

🌍 Live Geospatial Mapping: Interactive Leaflet maps featuring heatmaps, 5km buffer zones, and livestock trade corridor exposure tracking to visualize downstream spread risk.

📊 Dynamic CSV Auto-Adaptation: Integrated CSV file upload directly in the dashboard hotbar. The internal ML model re-parses the uploaded data and instantly adapts the dashboard without restarting the server.

🌤️ Live Climate Telemetry: Fetches real-time temperature spikes and rainfall deficits via the Open-Meteo API to dynamically adjust environmental risk scores.

🎛️ Fully Reactive Dashboard & "What-If" Engine: Real-time expert calibration sliders for disease indicator weights and a ring-vaccination "What-If" simulator that instantly recalculates risk projections.

📈 Interactive Sparkline Graphs: Click and drag across the risk-trend sparkline to dynamically highlight temporal regions and manually alter risk values on the fly.

🌐 Complete Multilingual Support: Instantly translate the entire UI, alerts, and downloadable Situation Reports into English, Hindi, and Marathi.

🚨 48-Hour SLA Response Queue & Dispatch: Tracks critical alerts with an integrated timeline. Features a webhook integration to push simulated automated WhatsApp/SMS dispatches to DVOs.

📱 Field Report Intake: Live mobile-friendly intake form for field veterinarians to log abortions, deaths, and PPR/FMD cases, immediately updating the model and map.

🛠️ Tech Stack

Backend: Python 3, Pandas, NumPy, Scikit-learn (Isolation Forest)

Frontend: HTML5, CSS3, Vanilla JavaScript, SVG

Mapping & Viz: Leaflet.js, Leaflet-heat.js

Server / APIs: Built-in Python http.server, Open-Meteo API

Deployment: Render.com (Continuous Deployment)

🚀 How to Run Locally

Clone the repository:

git clone https://github.com/your-username/spillover-radar.git
cd spillover-radar


Install the dependencies:
Make sure you have Python 3 installed.

pip install -r requirements.txt


Start the server:
To run the app with standard simulated data:

python run.py


To run the app with live Open-Meteo climate data:

python run.py --live


View the Dashboard:
The script will automatically open your default browser. If not, navigate to http://localhost:8000/index.html.

📂 Project Structure

run.py - Core application (Backend HTTP API server, ML logic, HTML template).

requirements.txt - Python dependencies (pandas, numpy, scikit-learn).

data/surveillance.csv - Auto-generated sample dataset for the dashboard.

api/districts.json - Computed dashboard payload for the frontend.

🤝 Future Roadmap

Automated GenAI SitReps: Integrating an LLM endpoint that automatically compiles current critical alerts, climate anomalies, and livestock data into a downloadable formal PDF/text SitRep for health ministers.

Compartmental Disease Spread Simulation (SIR Model): Extending the corridor exposure index into a full epidemiological network simulation to visualize how a pathogen travels across specific transport links over time.

Offline-First Field Mobile View: A lightweight mobile-responsive tab where field veterinarians can snap photos, record geo-tagged voice notes, and instantly push live reports.

Smart Vaccine & Resource Allocation Optimizer: An optimization module where users input a limited pool of rapid-response teams, and the algorithm computes the best deployment targets to minimize regional spread.
