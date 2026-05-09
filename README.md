# Sahwa Monitoring System v2.0

This project was developed at Abu Dhabi University (URIC 2026). It features a real-time dashboard for monitoring sensor data (wrist and ankle) and performing live inference using machine learning models.

## Deployment

The project is ready for deployment on platforms like Railway, Heroku, or locally.

### Local Setup & Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YoussefTheCreator-dev/sahwa-dashboard.git
   cd sahwa-dashboard
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the dashboard:**
   ```bash
   python sahwa_server.py
   ```

4. **Access the dashboard:**
   Open your browser and go to `http://127.0.0.1:5000`

## Features

- **Real-time TCP Listener:** Receives sensor data on port `8888`.
- **Live Inference:** Uses pre-trained Random Forest models for activity detection.
- **Interactive Dashboard:** Visualizes sensor streams and detected episodes.
- **Data Collection:** Built-in tools for collecting and labeling new sensor data.

## Deployment on Railway

This project includes a `Procfile` and `requirements.txt`, making it ready for one-click deployment on Railway. Simply connect your GitHub repository to a new Railway project.
