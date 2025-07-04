# Spot On
# Spot On: A blend of GIS and LLMs(Gemini)

Spot On is an advanced, AI-assisted land suitability analysis platform. Built with **Streamlit**, **PostGIS**, **PyLUSAT**, and **Google Gemini**, it identifies suitable locations using user defined spatial criteria, proximity analysis, and dynamic AI reasoning. The idea was to blend GIS with AI and create something thateven non experts can use easily. 

> “Find the best spot, based on your thoughts.”

---


## 🚀 About Spot On
![Screenshot 2025-04-28 181740](https://github.com/user-attachments/assets/762beb02-97e0-413d-89ef-06349fee088a)

---

## 🚀 Features

- 📍 **Multi-Criteria Weighted Overlay Analysis**
- 🧠 **LLM-Powered Prompt Parsing** (via Google Gemini)
- 📊 **Distance-Based Raster Processing**
- 🧾 **PDF Report Generation with Maps & Explanations**
- 🗂️ **Layer Management from PostGIS Database**
- 🧭 **Top Site Extraction with Real-World Geocoding**
- 🌐 **Interactive Leaflet Maps (via Folium)**
- 🖼️ **Matplotlib Suitability Visualizations**

---


## 🛠️ Tech Stack

- **Backend**: Python, Streamlit, GeoPandas, Rasterio, SQLAlchemy
- **Database**: PostgreSQL + PostGIS
- **AI Engine**: Google Gemini (`gemini-2.0-flash`)
- **Spatial Processing**: PyLUSAT, GDAL
- **Visualization**: Matplotlib, Folium, ReportLab
- **Geocoding**: Nominatim (OpenStreetMap)

---




---

### 💬 Define Criteria Using Prompt
![Prompt Parse](screenshots/a.png)

### 🧠 Identified Spatial Layers (AI Parsed)
![Identified Criteria](screenshots/b.png)

### 🗃️ Loading Spatial Data from Database
![loading data](screenshots/c.png)

### ⚙️ Analysis Result (Weighted Overlay)
![Analysis Result](screenshots/d.png)

### 🗺️ Suitability Map Output
![Suitability Map](screenshots/e.png)

### 📍 Top Suitable Locations (Ranked)
![Suitable Locations](screenshots/f.png)

### 🌐 Interactive Leaflet Map
![Interactive Map](screenshots/g.png)

### 🤖 AI-Powered Site Analysis
![AI Analysis](screenshots/h.png)

### 🏆 Top 3 Recommended Locations
![Nearby Features](screenshots/i.png)

### 🏘️ Nearby Features for Selected Site
![Nearby Features](screenshots/j.png)

### 🧭 More Nearby Features (Reverse Geocoded)
![Nearby Features](screenshots/k.png)

### 🧠 Gemini Explanation Summary
![Explanation](screenshots/l.png)

---



## 🎥 Demo Video

[▶️ Watch the Demo on YouTube](https://youtu.be/9YN-82Jp1Rs)

---
>

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/Noor-e-Fatima/spot-on
cd spot-on

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt















