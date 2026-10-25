"""
Spillover Radar  — One Health Surveillance Dashboard
Hackathon: Hack4Pashudhan| Team ID: H4P-VC587GBK
Team: VetTech Trio (Parth Kantharia, Priyal Kantharia, Rahul Bhojkar)
"""

import argparse
import datetime as dt
import http.server
import json
import os
import socketserver
import sys
import threading
import time
import urllib.request
import webbrowser
import hashlib
import io

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HUMAN_LAG_WEEKS = 2
HUMAN_ALERT = 3
REF_CAP = np.array([60.0, 10.0, 10.0])
WEIGHTS = np.array([0.50, 0.25, 0.25])
FACTOR_NAMES = ["Livestock Deaths, Abortions & PPR",
                "Thermal Shock & Rainfall Deficit", "Vaccination Coverage Deficit"]
STEPS = ["Flagged", "Verified", "Team dispatched", "Samples sent", "Closed"]
REQ_COLS = ["name", "state", "lat", "lon", "week", "abortions", "deaths", "ppr", "vax_cov",
            "temp_spike", "rain_deficit", "human_fever"]
CSV_PATH = os.path.join(BASE_DIR, "data", "surveillance.csv")

# --------------------------------------------------------------------------
# 1. [D.02] MULTI-SECTOR DATA INGESTION
# --------------------------------------------------------------------------
DISTRICTS = [
    ("Barmer", "Rajasthan", 25.7532, 71.3967, 8, 5, 14, 0.32, 3.1, 46, 0.35),
    ("Jodhpur", "Rajasthan", 26.2389, 73.0243, 5, 3, 9, 0.41, 2.4, 38, 0.60),
    ("Bikaner", "Rajasthan", 28.0229, 73.3119, 6, 4, 11, 0.38, 2.7, 40, 0.55),
    ("Jaisalmer", "Rajasthan", 26.9157, 70.9083, 5, 3, 8, 0.44, 2.9, 44, 0.50),
    ("Jaipur", "Rajasthan", 26.9124, 75.7873, 3, 1, 4, 0.63, 1.6, 22, 0.80),
    ("Kutch", "Gujarat", 23.2420, 69.6669, 4, 2, 7, 0.52, 2.2, 31, 0.65),
    ("Ahmednagar", "Maharashtra", 19.0948, 74.7480, 3, 2, 3, 0.66, 1.4, 18, 0.85),
    ("Pune", "Maharashtra", 18.5204, 73.8567, 2, 1, 0, 0.74, 1.1, 12, 1.00),
    ("Latur", "Maharashtra", 18.4088, 76.5604, 3, 1, 5, 0.58, 1.8, 26, 0.75),
    ("Bhopal", "Madhya Pradesh", 23.2599, 77.4126, 2, 1, 2, 0.71, 1.2, 14, 0.95),
    ("Wayanad", "Kerala", 11.6854, 76.1320, 1, 0, 1, 0.89, 0.3, -4, 1.00),
    ("Kozhikode", "Kerala", 11.2588, 75.7804, 1, 1, 1, 0.86, 0.5, 2, 1.05),
]
CORRIDORS = [
    ("Barmer", "Jodhpur", "Marwar cattle route", 0.9),
    ("Barmer", "Jaisalmer", "Thar border route", 0.5),
    ("Barmer", "Kutch", "Kutch–Barmer trade route", 0.6),
    ("Jodhpur", "Bikaner", "Western Rajasthan fair route", 0.6),
    ("Jodhpur", "Jaipur", "Jaipur mandi route", 0.9),
    ("Jaipur", "Bhopal", "Central India transit", 0.7),
    ("Pune", "Ahmednagar", "Deccan route", 0.8),
    ("Ahmednagar", "Latur", "Marathwada route", 0.6),
    ("Bhopal", "Ahmednagar", "Satpura transit", 0.5),
    ("Kozhikode", "Wayanad", "Malabar hill route", 0.4),
]
WEEKS_DEMO = 8


def build_demo_data(weeks=WEEKS_DEMO, start_scale=1.0):
    rng = np.random.RandomState(42)
    rows = []
    for (name, state, lat, lon, ab, de, pp, vc, ts, rd, start) in DISTRICTS:
        ramp = np.linspace(start * start_scale if start <
                           0.9 else start, 1.0, weeks)
        vet_hist = []
        for w in range(weeks):
            f = ramp[w]
            noise = (lambda: 1 + rng.normal(0, 0.08)
                     ) if w < weeks - 1 else (lambda: 1.0)
            a = max(0, int(round(ab * f * noise())))
            d = max(0, int(round(de * f * noise())))
            p = max(0, int(round(pp * f * noise())))
            t = round(ts * f * noise(), 2)
            r = round(rd * f * noise(), 1)
            vet = a * 2.5 + d * 4.0 + p * 1.5
            vet_hist.append(vet)
            human = max(0, int(
                round(0.28 * vet_hist[max(w - HUMAN_LAG_WEEKS, 0)] + rng.normal(0, 0.6))))
            rows.append(dict(name=name, state=state, lat=lat, lon=lon, week=w, abortions=a, deaths=d, ppr=p,
                             vax_cov=vc, temp_spike=t, rain_deficit=r, human_fever=human))
    return pd.DataFrame(rows)


def fetch_open_meteo(df):
    d = df.drop_duplicates("name")
    url = ("https://api.open-meteo.com/v1/forecast?latitude=" + ",".join(f"{x:.3f}" for x in d["lat"]) +
           "&longitude=" + ",".join(f"{x:.3f}" for x in d["lon"]) +
           "&daily=temperature_2m_max,precipitation_sum&past_days=35&forecast_days=1&timezone=auto")
    with urllib.request.urlopen(url, timeout=12) as r:
        data = json.loads(r.read())
    data = data if isinstance(data, list) else [data]
    out = {}
    for nm, item in zip(d["name"], data):
        tmax = np.array(
            [v if v is not None else np.nan for v in item["daily"]["temperature_2m_max"]], float)
        rain = np.array(
            [v if v is not None else 0.0 for v in item["daily"]["precipitation_sum"]], float)
        spike = float(np.nanmean(tmax[-7:]) - np.nanmean(tmax[-35:-7]))
        recent, prior = rain[-14:].sum(), rain[-35:-14].sum() * 14 / 21
        deficit = float(np.clip((prior - recent) / prior *
                        100, -100, 100)) if prior > 1 else 0.0
        out[nm] = (round(max(spike, 0.0), 2), round(deficit, 1))
    return out


STATE = {"scenario": None, "base": None, "payload": None, "alerts": {}, "events": [], "sources": [], "live": False,
         "climate_mode": "simulated", "last_refresh": None}
LOCK = threading.RLock()


def log_event(text):
    STATE["events"].insert(0, {"ts": dt.datetime.now(
    ).astimezone().isoformat(timespec="seconds"), "text": text})
    del STATE["events"][40:]


def apply_live_climate():
    try:
        res = fetch_open_meteo(STATE["base"])
        with LOCK:
            b = STATE["base"]
            last = b["week"].max()
            for nm, (ts, rd) in res.items():
                m = (b["name"] == nm) & (b["week"] == last)
                b.loc[m, "temp_spike"] = ts
                b.loc[m, "rain_deficit"] = rd
            STATE["climate_mode"] = f"live · Open-Meteo ({len(res)} districts)"
            STATE["last_refresh"] = dt.datetime.now(
            ).astimezone().isoformat(timespec="seconds")
            log_event(f"Live climate refreshed for {len(res)} districts")
        return True
    except Exception as e:
        with LOCK:
            STATE["climate_mode"] = "simulated (live fetch failed)"
            log_event(f"Live climate fetch failed: {type(e).__name__}")
        print(f"[warn] live climate fetch failed: {e}")
        return False


def load_base(live, scenario=None):
    path = scenario or CSV_PATH
    STATE["scenario"] = None
    if scenario or os.path.exists(CSV_PATH):
        d = pd.read_csv(path)
        miss = [c for c in REQ_COLS if c not in d.columns]
        if miss:
            raise SystemExit(f"{path} is missing columns: {miss}")
        detail = f"{d['name'].nunique()} districts x {d['week'].nunique()} weeks"
        if scenario:
            mp = os.path.splitext(path)[0] + ".json"
            meta = json.load(open(
                mp, encoding="utf-8")) if os.path.exists(mp) else {"title": os.path.basename(path)}
            STATE["scenario"] = meta
            mode = "scenario \u00b7 " + meta.get("title", "replay")
        else:
            mode = "CSV \u00b7 data/surveillance.csv"
    else:
        d = build_demo_data()
        mode, detail = "simulated", "synthetic demo series"
        os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
        d.to_csv(os.path.join(BASE_DIR, "data",
                 "surveillance_sample.csv"), index=False)
        gap = d["name"].isin(["Kutch", "Latur"]) & (
            d["week"] == d["week"].max())
        d.loc[gap, ["abortions", "deaths", "ppr"]] = np.nan
    STATE["base"] = d.reset_index(drop=True)
    STATE["sources"] = [("Animal health", mode, detail, "NADRES / state AH dept"),
                        ("Human health", mode, detail, "IDSP / IHIP")]
    STATE["live"] = live
    if live:
        apply_live_climate()


def write_scenario_sample():
    d = build_demo_data(10, 0.3)
    folder = os.path.join(BASE_DIR, "data", "scenarios")
    os.makedirs(folder, exist_ok=True)
    p = os.path.join(folder, "illustrative_outbreak.csv")
    d.to_csv(p, index=False)
    meta = {"title": "Illustrative outbreak replay (SYNTHETIC)", "illustrative": True, "focus_district": "Barmer",
            "confirmed_week": 9, "source": "synthetic - replace with a documented case",
            "notes": "confirmed_week is 1-indexed."}
    json.dump(meta, open(os.path.splitext(
        p)[0] + ".json", "w", encoding="utf-8"), indent=2)
    return p


WEBHOOK = None


def notify(alert, score):
    if not WEBHOOK:
        return

    def go():
        try:
            body = json.dumps({"text": f"SPILLOVER RADAR ALERT: {alert['district']} CRITICAL ({score}/100).",
                               "district": alert["district"], "score": score, "opened": alert["opened"]}).encode()
            urllib.request.urlopen(urllib.request.Request(WEBHOOK, data=body, headers={
                                   "Content-Type": "application/json"}), timeout=8)
        except Exception as e:
            print("[warn] webhook failed:", e)
    threading.Thread(target=go, daemon=True).start()

# --------------------------------------------------------------------------
# 2. [D.03] SCORING, ANOMALY DETECTION, ATTRIBUTION
# --------------------------------------------------------------------------


def cusum_week(series, k=0.5, h=4.0, base_n=3):
    mu, sd = np.mean(series[:base_n]), max(np.std(series[:base_n]), 1.0)
    s = 0.0
    for t, x in enumerate(series):
        s = max(0.0, s + (x - mu) / sd - k)
        if t >= base_n and s > h:
            return t
    return None


def compute_payload():
    with LOCK:
        base = STATE["base"].copy()
        climate_mode, live, last_refresh = STATE["climate_mode"], STATE["live"], STATE["last_refresh"]
        srcs = list(STATE["sources"])
    n = int(base["week"].max()) + 1
    df = base.sort_values(
        ["name", "week"], kind="stable").reset_index(drop=True)
    order = list(dict.fromkeys(base["name"]))

    numc = ["abortions", "deaths", "ppr",
            "human_fever", "temp_spike", "rain_deficit"]
    rec = df[df["week"] >= max(0, n - 3)]
    comp = ((1 - rec[numc].isna().groupby(rec["name"]
                                          ).mean().mean(axis=1)) * 100).round()
    gap = df[df["week"] == n -
             1].set_index("name")[["abortions", "deaths", "ppr"]].isna().any(axis=1)
    df[numc] = df.groupby("name")[numc].transform(
        lambda x: x.ffill()).fillna(0)

    df["vet"] = df["abortions"] * 2.5 + df["deaths"] * 4.0 + df["ppr"] * 1.5
    df["clim"] = df["temp_spike"] * 1.8 + \
        df["rain_deficit"].clip(lower=0) / 15.0
    df["vd"] = (1.0 - df["vax_cov"]) * 10.0
    F = df[["vet", "clim", "vd"]].values
    contrib = np.clip(F / REF_CAP, 0, 1) * WEIGHTS
    df["risk"] = np.round(contrib.sum(1) * 100, 1)
    tot = contrib.sum(1, keepdims=True)
    shares = contrib / np.where(tot == 0, 1, tot)

    sc = StandardScaler().fit(F)
    Xs = sc.transform(F)
    iso = IsolationForest(
        n_estimators=300, contamination=0.12, random_state=42).fit(Xs)
    s0 = iso.score_samples(Xs)
    rawA = -s0
    df["ml_score"] = np.round(
        100 * (rawA - rawA.min()) / max(rawA.max() - rawA.min(), 1e-9), 0)
    pred = iso.predict(Xs) == -1
    med = np.median(Xs, axis=0)
    gain = np.zeros_like(Xs)
    for j in range(3):
        Xj = Xs.copy()
        Xj[:, j] = med[j]
        gain[:, j] = np.clip(iso.score_samples(Xj) - s0, 0, None)
    gsum = gain.sum(1, keepdims=True)
    mlsh = gain / np.where(gsum == 0, 1, gsum)

    latest_mask = (df["week"] == n - 1).values
    cohort_mean = df.loc[latest_mask, "risk"].mean()
    df["ml_flag"] = pred & (df["risk"].values > cohort_mean) & latest_mask

    districts = []
    for nm in order:
        idx = np.where(df["name"].values == nm)[0]
        g = df.iloc[idx]
        cw = cusum_week(g["vet"].values)

        seed = int(hashlib.md5(nm.encode()).hexdigest(), 16)
        teams = (seed % 4) + 1
        vaccines = ((seed % 20) + 5) * 1500

        districts.append({
            "id": nm.lower(), "name": nm, "state": g["state"].iloc[0],
            "lat": float(g["lat"].iloc[0]), "lon": float(g["lon"].iloc[0]),
            "vet": [round(float(x), 2) for x in g["vet"]], "clim": [round(float(x), 2) for x in g["clim"]],
            "vax_cov": float(g["vax_cov"].iloc[-1]),
            "scores": [float(x) for x in g["risk"]],
            "human": [int(x) for x in g["human_fever"]], "deaths": [int(x) for x in g["deaths"]],
            "abortions": [int(x) for x in g["abortions"]], "ppr": [int(x) for x in g["ppr"]],
            "temp": [float(x) for x in g["temp_spike"]], "rain": [float(x) for x in g["rain_deficit"]],
            "shares": [[int(round(v * 100)) for v in shares[i]] for i in idx],
            "ml_score": [int(x) for x in g["ml_score"]],
            "ml_shares": [int(round(v * 100)) for v in mlsh[idx[-1]]],
            "ml_anomaly": bool(g["ml_flag"].iloc[-1]), "cusum_week": cw,
            "complete": int(comp[nm]), "gap": bool(gap[nm]),
            "teams": teams, "vaccines": vaccines
        })

    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    with LOCK:
        for d in districts:
            if d["scores"][-1] >= 70 and d["id"] not in STATE["alerts"]:
                STATE["alerts"][d["id"]] = {
                    "district": d["name"], "id": d["id"], "opened": now, "step": 0}
                log_event(
                    f"ALERT opened: {d['name']} reached {d['scores'][-1]}")
                notify(STATE["alerts"][d["id"]], d["scores"][-1])
        alerts, events = list(STATE["alerts"].values()), list(STATE["events"])

    climate_updated = last_refresh or "n/a"
    sources = [{"sector": s, "mode": m, "detail": d, "feed": f} for s, m, d, f in srcs] + \
              [{"sector": "Climate", "mode": climate_mode,
                  "detail": f"last refresh {climate_updated}", "feed": "IMD / Open-Meteo"}]
    payload = {
        "generated_at": now, "weeks": [f"Wk {w + 1}" for w in range(n)], "factorNames": FACTOR_NAMES,
        "weights": [float(x) for x in WEIGHTS], "ref_cap": [float(x) for x in REF_CAP],
        "districts": districts, "steps": STEPS, "alerts": alerts, "events": events, "sources": sources, "live": live,
        "corridors": [{"from": a.lower(), "to": b.lower(), "name": nm, "vol": v} for a, b, nm, v in CORRIDORS]
    }
    payload["webhook"] = bool(WEBHOOK)
    scen = STATE.get("scenario")
    payload["scenario"] = dict(scen) if scen else None

    with LOCK:
        STATE["payload"] = payload
    os.makedirs(os.path.join(BASE_DIR, "api"), exist_ok=True)
    with open(os.path.join(BASE_DIR, "api", "districts.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    return payload


# --------------------------------------------------------------------------
# 3. [D.01] DASHBOARD HTML
# --------------------------------------------------------------------------
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Spillover Radar | One Health Platform</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
<!-- SheetJS seamlessly handles Excel (.xlsx/.xls) and CSV parsing on the frontend -->
<script src="https://unpkg.com/xlsx/dist/xlsx.full.min.js"></script>
<style>
 :root {
   --bg-deep: #050714;
   --bg-panel: #0a0d24;
   --bg-card: rgba(17, 22, 51, 0.78);
   --border: rgba(192, 132, 252, 0.22);
   --purple: #c084fc;
   --pink: #f472b6;
   --text-main: #f5f3ff;
   --text-muted: #a5b4fc;
 }
 *{box-sizing:border-box}
 body{
   margin:0;
   font-family:'Inter',-apple-system,'Segoe UI',sans-serif;
   background-color: var(--bg-deep);
   background-image: 
     radial-gradient(circle at 15% 20%, rgba(192, 132, 252, 0.12) 0%, transparent 40%),
     radial-gradient(circle at 85% 80%, rgba(244, 114, 182, 0.10) 0%, transparent 40%),
     radial-gradient(rgba(192, 132, 252, 0.14) 1px, transparent 1px);
   background-size: 100% 100%, 100% 100%, 22px 22px;
   color: var(--text-main);
 }

 /* --- WELCOME SCREEN --- */
 #welcome {
   position: fixed;
   inset: 0;
   z-index: 99999;
   background-color: #040612;
   background-image: 
     radial-gradient(circle at 50% 35%, rgba(192, 132, 252, 0.20) 0%, rgba(244, 114, 182, 0.08) 35%, transparent 70%),
     radial-gradient(rgba(244, 114, 182, 0.16) 1px, transparent 1px);
   background-size: 100% 100%, 26px 26px;
   display: flex;
   align-items: center;
   justify-content: center;
   padding: 24px;
   transition: opacity 0.45s ease, transform 0.45s ease;
 }
 #welcome.hide {
   opacity: 0;
   transform: scale(1.04);
   pointer-events: none;
 }
 .w-card {
   max-width: 650px;
   width: 100%;
   background: rgba(10, 13, 36, 0.82);
   backdrop-filter: blur(16px);
   border: 1px solid rgba(192, 132, 252, 0.35);
   border-radius: 20px;
   padding: 42px 34px;
   text-align: center;
   box-shadow: 0 24px 60px rgba(0,0,0,0.75), 0 0 40px rgba(192, 132, 252, 0.15);
 }
 .w-radar {
   width: 78px;
   height: 78px;
   margin: 0 auto 20px;
   border-radius: 50%;
   background: radial-gradient(circle, rgba(244,114,182,0.35), rgba(192,132,252,0.08));
   border: 2px solid var(--pink);
   display: flex;
   align-items: center;
   justify-content: center;
   font-size: 34px;
   position: relative;
   box-shadow: 0 0 25px rgba(244,114,182,0.4);
 }
 .w-radar::after {
   content: '';
   position: absolute;
   inset: -8px;
   border-radius: 50%;
   border: 2px solid var(--purple);
   animation: pr 2s ease-out infinite;
 }
 .w-title {
   font-size: 40px;
   font-weight: 800;
   margin: 14px 0 10px;
   background: linear-gradient(90deg, #f5f3ff, #e879f9, #f472b6);
   -webkit-background-clip: text;
   -webkit-text-fill-color: transparent;
   letter-spacing: -0.5px;
 }
 .w-sub {
   color: #cbd5e1;
   font-size: 15px;
   line-height: 1.6;
   max-width: 520px;
   margin: 0 auto 24px;
 }
 .w-pills {
   display: flex;
   gap: 10px;
   justify-content: center;
   flex-wrap: wrap;
   margin-bottom: 30px;
 }
 .w-pill {
   background: rgba(192, 132, 252, 0.12);
   border: 1px solid rgba(192, 132, 252, 0.3);
   color: #e9d5ff;
   padding: 7px 14px;
   border-radius: 999px;
   font-size: 12px;
   font-weight: 600;
 }
 .w-btn {
   background: linear-gradient(135deg, #a855f7, #ec4899);
   color: #fff;
   border: none;
   padding: 14px 38px;
   font-size: 16px;
   font-weight: 700;
   border-radius: 999px;
   cursor: pointer;
   box-shadow: 0 10px 28px rgba(236, 72, 153, 0.45);
   transition: transform 0.2s, box-shadow 0.2s;
 }
 .w-btn:hover {
   transform: translateY(-2px) scale(1.02);
   box-shadow: 0 14px 34px rgba(236, 72, 153, 0.65);
 }

 /* --- HEADER & LAYOUT --- */
 header{
   background: rgba(10, 13, 36, 0.9);
   backdrop-filter: blur(10px);
   padding:8px 18px;
   border-bottom:1px solid var(--border);
   display:flex;justify-content:space-between;align-items:center;min-height:56px;gap:10px;flex-wrap:wrap
 }
 .badge{
   background: linear-gradient(135deg, #9333ea, #db2777);
   color: #fff;
   padding:4px 10px;border-radius:6px;font-size:11px;font-weight:bold;
   box-shadow: 0 2px 10px rgba(219, 39, 119, 0.35);
 }
 .chip{display:inline-block;padding:2px 8px;border-radius:10px;font-size:10px;font-weight:700;margin-left:5px;color:#050714}
 .live{background:#34d399}.csv{background:#c084fc}.sim{background:#f472b6}
 #container{display:flex;height:calc(100vh - 62px)}
 #mapwrap{flex:1;display:flex;flex-direction:column;min-width:0;position:relative}
 #controls{
   background: rgba(10, 13, 36, 0.88);
   border-bottom:1px solid var(--border);
   padding:9px 14px;display:flex;gap:14px;align-items:center;flex-wrap:wrap;font-size:12px;color:#e9d5ff
 }
 #controls label{
   cursor:pointer;display:flex;align-items:center;gap:5px;
   padding: 4px 9px;
   border-radius: 6px;
   background: rgba(192, 132, 252, 0.08);
   border: 1px solid rgba(192, 132, 252, 0.18);
   transition: all 0.15s;
 }
 #controls label:hover {
   border-color: var(--pink);
   background: rgba(244, 114, 182, 0.14);
 }
 #controls input[type=checkbox] { accent-color: var(--pink); }
 #map{flex:1; background: #040611;}
 #panel{
   width:430px;
   background: rgba(10, 13, 36, 0.86);
   backdrop-filter: blur(12px);
   border-left:1px solid var(--border);
   padding:12px;overflow-y:auto
 }
 @media(max-width:900px){#container{flex-direction:column;height:auto}#map{height:55vh;flex:none}#panel{width:100%}}
 .card{
   background: var(--bg-card);
   backdrop-filter: blur(8px);
   border-radius:10px;padding:12px;margin-bottom:10px;
   border:1px solid var(--border);
   box-shadow: 0 4px 20px rgba(0,0,0,0.35);
 }
 .card h4{margin:0 0 8px;font-size:11px;color:var(--pink);letter-spacing:.6px;text-transform:uppercase}
 .btn{
   background: linear-gradient(135deg, #a855f7, #ec4899);
   color:#fff;border:none;padding:6px 12px;border-radius:6px;cursor:pointer;font-weight:600;font-size:12px;
   transition: opacity 0.15s, transform 0.15s;
 }
 .btn:hover{opacity:0.92;transform:translateY(-1px)}
 .btn.alt{background:#191e42;border:1px solid rgba(192,132,252,0.3);color:#e9d5ff}
 .btn.ok{background:#10b981;color:#06231a}.btn:disabled{opacity:.4;cursor:default}
 .kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:10px}
 .kpi{background:var(--bg-card);border:1px solid var(--border);border-radius:8px;padding:8px;text-align:center}
 .kpi b{display:block;font-size:20px;color:#f472b6}.kpi span{font-size:10px;color:#d8b4fe}
 table{width:100%;border-collapse:collapse;font-size:12px}
 td,th{padding:5px 4px;text-align:left;border-bottom:1px solid rgba(192,132,252,0.14)}
 th{color:var(--purple);font-size:11px}
 tr.row{cursor:pointer}tr.row:hover{background:rgba(192,132,252,0.12)}tr.sel{background:rgba(244,114,182,0.2)}
 .pill{padding:1px 7px;border-radius:10px;font-size:10px;font-weight:700;color:#050714}
 .bar{height:7px;border-radius:4px;background:#191e42;overflow:hidden;margin:3px 0 6px}.bar i{display:block;height:100%}
 .note{font-size:11px;color:var(--text-muted);margin-top:6px;line-height:1.4}
 .flow{animation:dash 1.2s linear infinite}@keyframes dash{to{stroke-dashoffset:-20}}
 .dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}
 input[type=range]{width:100%;accent-color:var(--pink)}
 input[type=number],select{background:#070a1c;color:#f5f3ff;border:1px solid var(--border);border-radius:5px;padding:5px;width:100%;font-size:12px}
 .g3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px}
 .q{padding:7px 0;border-top:1px solid var(--border);font-size:12px}.q:first-child{border-top:0}
 .steps{display:flex;gap:3px;margin:4px 0}.steps i{flex:1;height:5px;border-radius:3px;background:#191e42}.steps i.on{background:var(--pink)}
 .mkr{position:relative;width:var(--s);height:var(--s);border-radius:50%;background:var(--c);border:2px solid #fff;color:#050714;font:800 11px/calc(var(--s) - 4px) system-ui,sans-serif;text-align:center;box-shadow:0 1px 8px rgba(0,0,0,.75)}
 .mkr.sel{outline:3px solid var(--pink);outline-offset:2px}
 .mkr.crit::after{content:'';position:absolute;inset:-2px;border-radius:50%;border:2px solid var(--c);animation:pr 1.6s ease-out infinite}
 @keyframes pr{from{transform:scale(1);opacity:.9}to{transform:scale(2.6);opacity:0}}
 .mkl{position:absolute;top:100%;left:50%;transform:translateX(-50%);white-space:nowrap;font:700 10px system-ui,sans-serif;color:#f5f3ff;text-shadow:0 0 3px #000,0 0 3px #000;margin-top:3px;pointer-events:none}
 .mtip{background:#0a0d24;color:#f5f3ff;border:1px solid var(--purple);font-size:12px;line-height:1.5}
 .leaflet-tooltip-top.mtip::before{border-top-color:#0a0d24}
 .maplegend,.mapchip{background:rgba(10,13,36,.92);color:#f5f3ff;padding:8px 10px;border-radius:8px;border:1px solid var(--border);font-size:11px;line-height:1.7}
 .mapchip{font-size:12px;font-weight:700;color:var(--pink)}
 #mapHint{
   position:absolute;top:56px;left:50%;transform:translateX(-50%);
   z-index:1000;background:rgba(10,13,36,0.92);border:1px solid var(--pink);
   color:#f5f3ff;padding:7px 16px;border-radius:999px;font-size:12px;font-weight:600;
   box-shadow:0 6px 20px rgba(0,0,0,0.6);pointer-events:none;transition:opacity 0.3s;
 }
 #banner{display:none;cursor:pointer;background:linear-gradient(90deg,#4c0519,#831843);color:#fbcfe8;padding:7px 18px;font-size:13px;font-weight:600;border-bottom:1px solid var(--pink)}
 #ac{display:none;position:fixed;inset:0;background:rgba(5,7,20,.78);backdrop-filter:blur(6px);z-index:5000;align-items:flex-start;justify-content:center;padding:40px 12px;overflow:auto}
 #bell.hot{background:#ec4899;color:#fff}
 #toast{position:fixed;bottom:18px;left:50%;transform:translateX(-50%);background:var(--pink);color:#050714;padding:8px 16px;border-radius:8px;font-weight:700;font-size:13px;display:none;z-index:9999}
 .sm{font-size:12px;color:#e9d5ff}
</style>
</head>
<body>

<!-- WELCOME SCREEN -->
<div id="welcome">
  <div class="w-card">
    <div class="w-radar">&#128225;</div>
    <div><span class="badge">Hack4Pashudhan &middot; Team ID: H4P-VC587GBK</span></div>
    <h1 class="w-title">Spillover Radar</h1>
    <p class="w-sub">
      One Health AI Surveillance &amp; Early Outbreak Detection Platform.<br>
      Built by <strong style="color:#f472b6">VetTech Trio</strong> (Parth Kantharia, Priyal Kantharia, Rahul Bhojkar).
    </p>
    <div class="w-pills">
      <span class="w-pill">&#129440; Zoonotic Risk Scoring</span>
      <span class="w-pill">&#127782;&#65039; Live Climate Anomaly</span>
      <span class="w-pill">&#129302; Isolation Forest + CUSUM</span>
      <span class="w-pill">&#128680; 48h Vet Dispatch Queue</span>
    </div>
    <button class="w-btn" onclick="startApp()">Get Started &rarr;</button>
  </div>
</div>

<header>
  <div>
    <span class="badge">Team ID: H4P-VC587GBK</span>
    <strong style="margin-left:10px;color:#f472b6">&#128225; Spillover Radar</strong>
    <span style="color:#c084fc;font-size:12px;margin-left:8px">VetTech Trio</span>
  </div>
  <div class="sm"><span id="fresh"></span><span id="chips"></span>
    <button class="btn alt" id="bell" onclick="openAC()" title="Alert Center">&#128276; <b id="bellN">0</b></button>
    <select id="lang" style="width:auto;margin-left:8px"><option value="en">EN</option><option value="hi">हिन्दी</option><option value="mr">मराठी</option></select>
    <button class="btn alt" style="margin-left:4px" onclick="refreshNow()">&#8635; <span data-i="refresh">Refresh</span></button>
    
    <button class="btn alt" style="margin-left:4px" onclick="importGoogleSheet()">&#128203; <span data-i="g_sheet">Google Sheet</span></button>
    <label class="btn alt" style="display:inline-block;cursor:pointer;margin-left:4px;padding:6px 12px;border-radius:6px;font-weight:600;font-size:12px;background:#191e42;border:1px solid rgba(192,132,252,0.3);color:#e9d5ff;">
      📁 <span data-i="excel_upload">Upload Excel/CSV</span>
      <input type="file" id="csvFileUp" accept=".csv, .xlsx, .xls" style="display:none" onchange="handleFileUpload(event)">
    </label>

    <button class="btn" style="margin-left:4px" onclick="sitrep()"><span data-i="sitrep">Situation report</span></button>
  </div>
</header>
<div id="scn" style="display:none;background:#4a044e;color:#f5d0fe;padding:6px 18px;font-size:12px"></div>
<div id="banner" onclick="openAC()"></div>
<div id="container">
  <div id="mapwrap">
    <div id="controls">
      <span style="color:#f472b6;font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:.5px" data-i="map_layers">Map Layers:</span>
      <label><input type="checkbox" id="lMarkers"> <span data-i="ly_markers">Markers</span></label>
      <label><input type="checkbox" id="lHeat"> <span data-i="ly_heat">Heatmap</span></label>
      <label><input type="checkbox" id="lBuf"> <span data-i="ly_buf">5 km buffers</span></label>
      <label><input type="checkbox" id="lCor"> <span data-i="ly_cor">Corridors</span></label>
      <label><input type="checkbox" id="lLab"> <span data-i="ly_lab">Labels</span></label>
      <span style="margin-left:auto"></span>
      <button class="btn alt" id="playBtn" onclick="togglePlay()">&#9654; <span data-i="play">Play</span></button>
      <input type="range" id="week" min="0" max="7" value="7" style="width:150px">
      <b id="weekLbl" style="min-width:42px;color:#f472b6"></b>
    </div>
    <div id="mapHint" data-i="map_hint">&#10024; Select layers above one by one or click a district on the right to customize details</div>
    <div id="map"></div>
  </div>
  <div id="panel">
    <div class="kpis">
      <div class="kpi"><b id="kCrit" style="color:#ef4444">0</b><span data-i="k_crit">CRITICAL</span></div>
      <div class="kpi"><b id="kWatch" style="color:#f59e0b">0</b><span data-i="k_watch">WATCH</span></div>
      <div class="kpi"><b id="kAnimal">0</b><span data-i="k_ad">ANIMAL DEATHS</span></div>
      <div class="kpi"><b id="kHuman">0</b><span data-i="k_hf">HUMAN FEVER</span></div>
    </div>

    <div class="card">
      <div style="display:flex; justify-content:space-between; align-items:flex-start;">
        <h3 id="dName" style="margin:0 0 4px;color:#f472b6"></h3>
        <span id="dOutlook" class="badge" style="font-size:10px;"></span>
      </div>
      <div id="dStatus" class="sm"></div>
      <div style="display:flex;align-items:baseline;gap:10px"><div id="dScore" style="font-size:32px;font-weight:bold;margin:6px 0 2px">--</div><div id="dWhat" class="sm"></div></div>
      <div id="dMeta" class="sm"></div><div id="dGap" class="sm" style="color:#f59e0b"></div>
      
      <div class="sm" style="margin-top:6px; padding:6px; background:rgba(192,132,252,0.1); border-radius:6px;">
        <b data-i="res_avl">Resources Available:</b> <span id="dTeams" style="color:#f472b6; font-weight:bold;">0</span> Vet Teams | <span id="dVax" style="color:#f472b6; font-weight:bold;">0</span> Vaccine Doses
      </div>

      <div style="position:relative; margin-top:6px;">
        <svg id="spark" viewBox="0 0 380 96" width="100%" style="cursor:crosshair; user-select:none;"></svg>
        <div id="graphTooltip" style="position:absolute; display:none; background:#0a0d24; border:1px solid var(--pink); color:#f5f3ff; font-size:10px; padding:2px 6px; border-radius:4px; pointer-events:none;"></div>
      </div>
      <div class="note" data-i="graph_note">Pink: risk &middot; drag or click across graph to adjust values &amp; highlight region &middot; purple dashed: human fever.</div>
      <div id="dTrend" class="sm" style="margin-top:4px"></div>
      
      <div style="margin-top:8px"><span class="sm"><span data-i="vax_whatif">What-if ring vaccination</span>: +<b id="vxv">0</b>% coverage</span>
        <input type="range" id="vx" min="0" max="40" value="0"></div>
      <div style="margin-top:8px;display:flex;gap:6px;flex-wrap:wrap">
        <button class="btn alt" onclick="zoomBuf()" data-i="zoom">Zoom to 5 km buffer</button>
        <button class="btn alt" onclick="dispatchDVO()" data-i="notify_dvo">&#128242; Notify DVO (WhatsApp)</button>
      </div>
    </div>

    <div class="card">
      <h4 data-i="why">RISK DRIVERS &amp; AI ATTRIBUTION</h4>
      <div id="dFactorsComb"></div>
      <div id="dDet" class="sm" style="margin-top:6px; padding-top:6px; border-top:1px dashed var(--border)"></div>
      <div class="note" data-i="attr_note">Combined view of indicator weights (Pink) and Isolation Forest anomaly attribution (Purple). Association, not causation.</div>
    </div>

    <div class="card"><h4 style="color:#f472b6" data-i="protocol">ACTIONABLE VETERINARY PROTOCOL</h4><p id="dAction" class="sm" style="margin:0"></p></div>

    <div class="card">
      <h4 data-i="queue_log">48-HOUR RESPONSE QUEUE &amp; ACTIVITY</h4>
      <div id="alerts"></div>
      <div style="border-top:1px solid var(--border); margin-top:8px; padding-top:8px;">
        <h4 data-i="log" style="margin-bottom:6px; color:#c084fc;">RECENT ACTIVITY</h4>
        <div id="events" class="sm" style="max-height:80px; overflow-y:auto; padding-right:4px;"></div>
      </div>
    </div>

    <div class="card">
      <h4 data-i="report">FIELD REPORT INTAKE (live)</h4>
      <select id="rDist"></select>
      <div class="g3" style="margin-top:6px">
        <input type="number" id="rAb" min="0" max="500" placeholder="Abortions">
        <input type="number" id="rDe" min="0" max="500" placeholder="Deaths">
        <input type="number" id="rPp" min="0" max="500" placeholder="PPR/FMD"></div>
      <button class="btn" style="margin-top:6px" onclick="sendReport()" data-i="submit">Submit report &amp; re-run pipeline</button>
      <div class="note" data-i="report_note">Adds to the latest week for that district. The model re-scores and the map updates immediately.</div>
    </div>

    <div class="card">
      <h4 data-i="expo">CORRIDOR EXPOSURE INDEX (downstream spread risk)</h4>
      <div id="exposure" class="sm"></div>
      <div class="note" data-i="expo_note">Origin risk &times; corridor volume &times; destination unvaccinated share. Volumes are illustrative until permit data is linked.</div>
    </div>

    <div class="card">
      <h4 data-i="calib">EXPERT CALIBRATION</h4>
      <select id="preset">
        <option value="0.5,0.25,0.25" data-i="p_gen">General syndromic</option>
        <option value="0.3,0.5,0.2" data-i="p_vec">Vector-borne / climate-driven (KFD-like)</option>
        <option value="0.4,0.15,0.45" data-i="p_vax">Vaccine-preventable (PPR/FMD-like)</option>
      </select>
      <div class="g3" style="margin-top:6px">
        <div class="sm"><span data-i="c_anim">Animal</span> <b id="w0"></b><input type="range" id="s0" min="0" max="100"></div>
        <div class="sm"><span data-i="c_clim">Climate</span> <b id="w1"></b><input type="range" id="s1" min="0" max="100"></div>
        <div class="sm"><span data-i="c_vax">Vaccine</span> <b id="w2"></b><input type="range" id="s2" min="0" max="100"></div></div>
      <div class="note" data-i="calib_note">Vets can retune weights live; scores, ranking, alerts view and exposure update. (Server-side alerts use default weights.)</div>
    </div>

    <div class="card"><h4><span data-i="rank">DISTRICT RANKING</span> &mdash; <span id="rankWk"></span></h4><table id="rank"></table></div>
    <div class="card">
      <span class="sm"><i class="dot" style="background:#ef4444"></i><span data-i="leg_crit">Critical &ge; 70</span></span>
      <span class="sm" style="margin-left:10px"><i class="dot" style="background:#f59e0b"></i><span data-i="leg_watch">Watch 40&ndash;69</span></span>
      <span class="sm" style="margin-left:10px"><i class="dot" style="background:#10b981"></i><span data-i="leg_norm">Normal &lt; 40</span></span>
      <div class="note" data-i="proto_note">Prototype decision-support tool. Scores are relative indicators, not calibrated probabilities or diagnoses. Confirm with laboratory and field investigation.</div>
    </div>
  </div>
</div>
<div id="ac" onclick="if(event.target===this)closeAC()">
 <div style="background:#0a0d24;border:1px solid var(--border);border-radius:12px;max-width:760px;width:100%;padding:16px">
  <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap">
   <h3 style="margin:0;color:#f472b6">&#128680; <span data-i="ac_title">Alert Center</span></h3>
   <div><button class="btn alt" onclick="simulate()" data-i="ac_sim">Simulate field-report surge</button>
        <button class="btn alt" onclick="askNotif()" data-i="ac_notif">Enable desktop notifications</button>
        <button class="btn alt" onclick="closeAC()">&#10005;</button></div></div>
  <div class="note" id="acHelp"></div>
  <div id="acList" style="margin-top:6px"></div>
  <h4 style="margin:12px 0 4px;font-size:11px;color:#f472b6" data-i="log">ACTIVITY LOG</h4><div id="acEvents" class="sm"></div>
 </div></div>
<div id="toast"></div>
<script>
const $ = id => document.getElementById(id);

function startApp() {
  const wel = $('welcome');
  if (wel) {
    wel.classList.add('hide');
    setTimeout(() => {
      wel.style.display = 'none';
      if (window.map) { 
        window.map.invalidateSize(); 
        if (typeof fitAll === 'function') fitAll(); 
      }
    }, 450);
  }
}

function updateHint() {
  const anyOn = ['lMarkers','lHeat','lBuf','lCor','lLab'].some(id => $(id) && $(id).checked);
  if ($('mapHint')) $('mapHint').style.opacity = anyOn ? '0' : '1';
}

const I18N = {
 en: { 
  crit:'CRITICAL', watch:'WATCH', normal:'NORMAL', status:'Status', refresh:'Refresh', sitrep:'Situation report', 
  k_crit:'CRITICAL', k_watch:'WATCH', k_ad:'ANIMAL DEATHS', k_hf:'HUMAN FEVER',
  why:'RISK DRIVERS & AI ATTRIBUTION', protocol:'ACTIONABLE VETERINARY PROTOCOL', queue_log:'48-HOUR RESPONSE QUEUE & ACTIVITY', 
  report:'FIELD REPORT INTAKE (live)', expo:'CORRIDOR EXPOSURE INDEX (downstream spread risk)',
  calib:'EXPERT CALIBRATION', log:'RECENT ACTIVITY', rank:'DISTRICT RANKING', 
  zoom:'Zoom to 5 km buffer', copy:'Copy DVO alert message', submit:'Submit report & re-run pipeline',
  g_sheet:'Google Sheet', excel_upload:'Upload Excel/CSV', map_layers:'Map Layers:', ly_markers:'Markers', ly_heat:'Heatmap', ly_buf:'5 km buffers', 
  ly_cor:'Corridors', ly_lab:'Labels', play:'Play', notify_dvo:'📲 Notify DVO (WhatsApp)', res_avl:'Resources Available:',
  map_hint:'✨ Select layers above one by one or click a district on the right to customize details',
  graph_note:'Pink: risk · drag or click across graph to adjust values & highlight region · purple dashed: human fever.',
  vax_whatif:'What-if ring vaccination',
  attr_note:'Combined view of indicator weights (Pink) and Isolation Forest anomaly attribution (Purple). Association, not causation.',
  report_note:'Adds to the latest week for that district. The model re-scores and the map updates immediately.',
  expo_note:'Origin risk × corridor volume × destination unvaccinated share. Volumes are illustrative until permit data is linked.',
  p_gen:'General syndromic', p_vec:'Vector-borne / climate-driven (KFD-like)', p_vax:'Vaccine-preventable (PPR/FMD-like)',
  c_anim:'Animal', c_clim:'Climate', c_vax:'Vaccine',
  calib_note:'Vets can retune weights live; scores, ranking, alerts view and exposure update. (Server-side alerts use default weights.)',
  leg_crit:'Critical ≥ 70', leg_watch:'Watch 40–69', leg_norm:'Normal < 40',
  proto_note:'Prototype decision-support tool. Scores are relative indicators, not calibrated probabilities or diagnoses. Confirm with laboratory and field investigation.',
  act_critical:'Deploy Veterinary Rapid Response within 48 h, verify reports, sample affected herds, restrict movement along exposed corridors, alert the District Health Officer, and consider targeted 5 km ring vaccination per disease protocol.',
  act_watch:'Increase syndromic surveillance, verify animal reports with block veterinary officers, review vaccination gaps, re-check in 48 h.', act_normal:'Routine syndromic monitoring.',
  st0:'Flagged', st1:'Verified', st2:'Team dispatched', st3:'Samples sent', st4:'Closed', left:'h left to dispatch', breach:'SLA breached', met:'SLA met', view:'View', adv:'Advance', noalert:'No open alerts.',
  compl:'Data completeness', gapwarn:'latest animal report missing - score uses last known values', a_alert:'SPILLOVER RADAR ALERT', a_risk:'risk', a_animal:'Animal', a_deaths:'Deaths', a_abort:'Abortions', a_human:'Human fever cases', a_link:'Linked districts',
  a_action:'Action', a_c:'verify and dispatch vet team within 48 h; inform DHO.', a_w:'verify reports; heightened surveillance.', a_n:'routine monitoring.', a_foot:'Decision-support only; confirm via lab/field.',
  'Livestock Deaths, Abortions & PPR': 'Livestock Deaths, Abortions & PPR', 'Thermal Shock & Rainfall Deficit': 'Thermal Shock & Rainfall Deficit', 'Vaccination Coverage Deficit': 'Vaccination Coverage Deficit'
 },
 hi: { 
  crit:'गंभीर', watch:'निगरानी', normal:'सामान्य', status:'स्थिति', refresh:'ताज़ा करें', sitrep:'स्थिति रिपोर्ट', 
  k_crit:'गंभीर', k_watch:'निगरानी', k_ad:'पशु मृत्यु', k_hf:'मानव बुखार',
  why:'जोखिम कारक और एआई एट्रिब्यूशन', protocol:'पशु चिकित्सा कार्य-योजना', queue_log:'48 घंटे की प्रतिक्रिया और गतिविधि', 
  report:'क्षेत्र रिपोर्ट दर्ज करें (लाइव)', expo:'गलियारा जोखिम सूचकांक (आगे फैलाव का खतरा)',
  calib:'विशेषज्ञ अंशांकन', log:'हाल की गतिविधि', rank:'जिला क्रम', notify_dvo:'📲 डीवीओ को सूचित करें (WhatsApp)', res_avl:'उपलब्ध संसाधन:',
  zoom:'5 किमी क्षेत्र दिखाएँ', copy:'जिला पशु चिकित्सा अधिकारी के लिए अलर्ट कॉपी करें', submit:'रिपोर्ट जमा करें',
  g_sheet:'गूगल शीट', excel_upload:'एक्सेल/CSV अपलोड करें', map_layers:'मानचित्र परतें:', ly_markers:'मार्कर', ly_heat:'हीटमैप', ly_buf:'5 किमी बफर', 
  ly_cor:'गलियारे', ly_lab:'लेबल', play:'चलाएँ',
  map_hint:'✨ विवरण अनुकूलित करने के लिए ऊपर दी गई परतों को एक-एक करके चुनें',
  graph_note:'गुलाबी: जोखिम · मान बदलने और क्षेत्र को हाइलाइट करने के लिए ग्राफ़ पर खींचें या क्लिक करें · बैंगनी डैश: मानव बुखार।',
  vax_whatif:'यदि रिंग टीकाकरण हो', 
  attr_note:'संकेतक भार (गुलाबी) और एट्रिब्यूशन (बैंगनी) का संयुक्त दृश्य।',
  report_note:'उस जिले के नवीनतम सप्ताह में जुड़ता है। मॉडल पुन: स्कोर करता है और नक्शा तुरंत अपडेट होता है।',
  expo_note:'मूल जोखिम × गलियारा वॉल्यूम × गंतव्य असंक्रमित हिस्सा।',
  p_gen:'सामान्य सिंड्रोमिक', p_vec:'वेक्टर-जनित / जलवायु-संचालित', p_vax:'टीका-रोकथाम योग्य',
  c_anim:'पशु', c_clim:'जलवायु', c_vax:'टीका',
  calib_note:'पशु चिकित्सक लाइव वजन बदल सकते हैं।',
  leg_crit:'गंभीर ≥ 70', leg_watch:'निगरानी 40–69', leg_norm:'सामान्य < 40',
  proto_note:'प्रोटोटाइप निर्णय-सहायता टूल। प्रयोगशाला जाँच से पुष्टि करें।',
  act_critical:'48 घंटे के भीतर पशु चिकित्सा रैपिड रेस्पॉन्स टीम भेजें, रिपोर्टों की पुष्टि करें, प्रभावित पशुओं के नमूने लें, जोखिम वाले गलियारों में पशु आवाजाही सीमित करें, जिला स्वास्थ्य अधिकारी को सूचित करें, और रोग-विशेष प्रोटोकॉल के अनुसार 5 किमी में लक्षित रिंग टीकाकरण पर विचार करें।',
  act_watch:'निगरानी बढ़ाएँ, खंड पशु चिकित्साधिकारी से रिपोर्टों की पुष्टि करें, टीकाकरण अंतर की समीक्षा करें और 48 घंटे में दोबारा जाँचें।', act_normal:'नियमित निगरानी जारी रखें।',
  st0:'चिह्नित', st1:'सत्यापित', st2:'टीम रवाना', st3:'नमूने भेजे', st4:'बंद', left:'घंटे शेष (टीम भेजने तक)', breach:'समय-सीमा चूकी', met:'समय-सीमा पूरी', view:'देखें', adv:'आगे बढ़ाएँ', noalert:'कोई खुला अलर्ट नहीं।',
  compl:'डेटा पूर्णता', gapwarn:'नवीनतम पशु रिपोर्ट अनुपलब्ध — स्कोर पिछले मान पर आधारित', a_alert:'स्पिलओवर रडार अलर्ट', a_risk:'जोखिम', a_animal:'पशु', a_deaths:'मृत्यु', a_abort:'गर्भपात', a_human:'मानव बुखार के मामले', a_link:'जुड़े जिले',
  a_action:'कार्रवाई', a_c:'48 घंटे में पशु चिकित्सा टीम भेजें; जिला स्वास्थ्य अधिकारी को सूचित करें।', a_w:'रिपोर्टों की पुष्टि करें; निगरानी बढ़ाएँ।', a_n:'नियमित निगरानी।', a_foot:'केवल निर्णय-सहायता; प्रयोगशाला/क्षेत्र जाँच से पुष्टि करें।',
  'Livestock Deaths, Abortions & PPR': 'पशु मृत्यु, गर्भपात और पीपीआर', 'Thermal Shock & Rainfall Deficit': 'थर्मल शॉक और वर्षा की कमी', 'Vaccination Coverage Deficit': 'टीकाकरण कवरेज की कमी'
 },
 mr: { 
  crit:'गंभीर', watch:'निरीक्षण', normal:'सामान्य', status:'स्थिती', refresh:'रिफ्रेश करा', sitrep:'परिस्थिती अहवाल', 
  k_crit:'गंभीर', k_watch:'निरीक्षण', k_ad:'पशू मृत्यू', k_hf:'मानवी ताप',
  why:'जोखीम चालक आणि एआय गुणधर्म', protocol:'पशुवैद्यकीय कृती योजना', queue_log:'48 तासांची प्रतिक्रिया आणि कृती', 
  report:'क्षेत्र अहवाल नोंदवा (लाइव्ह)', expo:'मार्गिका जोखीम निर्देशांक (पुढील प्रसार धोका)',
  calib:'तज्ज्ञ कॅलिब्रेशन', log:'अलीकडील क्रियाकलाप', rank:'जिल्हा क्रमवारी', notify_dvo:'📲 डीव्हीओला सूचित करा (WhatsApp)', res_avl:'उपलब्ध संसाधने:',
  zoom:'5 किमी क्षेत्र दाखवा', copy:'जिल्हा पशुवैद्यकीय अधिकाऱ्यासाठी अलर्ट कॉपी करा', submit:'अहवाल सादर करा',
  g_sheet:'गुगल शीट', excel_upload:'एक्सेल/CSV अपलोड करा', map_layers:'नकाशा स्तर:', ly_markers:'मार्कर्स', ly_heat:'हिटमॅप', ly_buf:'५ किमी बफर', 
  ly_cor:'मार्गिका', ly_lab:'लेबल्स', play:'प्ले',
  map_hint:'✨ तपशील अनुकूलित करण्यासाठी वरील स्तर एकामागून एक निवडा',
  graph_note:'गुलाबी: जोखीम · मूल्य बदलण्यासाठी आणि प्रदेश हाईलाइट करण्यासाठी आलेखावर ड्रॅग किंवा क्लिक करा.',
  vax_whatif:'रिंग लसीकरण झाल्यास', 
  attr_note:'निर्देशक वजन (गुलाबी) आणि गुणधर्म (जांभळा) यांचे संयुक्त दृश्य.',
  report_note:'त्या जिल्ह्याच्या ताज्या आठवड्यात जोडले जाते. मॉडेल पुनर्मूल्यांकन करते.',
  expo_note:'मूळ जोखीम × मार्गिका व्हॉल्यूम × गंतव्य विना-लसीकरण हिस्सा.',
  p_gen:'सामान्य सिंड्रोमिक', p_vec:'व्हेक्टर-जनित / हवामान-चालित', p_vax:'लसीकरण-प्रतिबंध योग्य',
  c_anim:'प्राणी', c_clim:'हवामान', c_vax:'लसीकरण',
  calib_note:'पशुवैद्य लाइव्ह वजन बदलू शकतात.',
  leg_crit:'गंभीर ≥ 70', leg_watch:'निरीक्षण 40–69', leg_norm:'सामान्य < 40',
  proto_note:'प्रोटोटाइप निर्णय-सहाय्य साधन. प्रयोगशाळा तपासणीने पुष्टी करा.',
  act_critical:'48 तासांत पशुवैद्यकीय जलद प्रतिसाद पथक पाठवा, अहवालांची खात्री करा, बाधित जनावरांचे नमुने घ्या, धोक्याच्या मार्गिकांवर जनावरांची ये-जा मर्यादित करा, जिल्हा आरोग्य अधिकाऱ्यांना कळवा आणि रोगनिहाय नियमावलीनुसार 5 किमी परिसरात लक्ष्यित रिंग लसीकरणाचा विचार करा.',
  act_watch:'निगराणी वाढवा, तालुका पशुवैद्यकीय अधिकाऱ्यांकडून अहवालांची खात्री करा, लसीकरणातील तफावत तपासा आणि 48 तासांत पुन्हा तपासा.', act_normal:'नियमित निगराणी सुरू ठेवा.',
  st0:'चिन्हांकित', st1:'पडताळणी झाली', st2:'पथक रवाना', st3:'नमुने पाठवले', st4:'बंद', left:'तास शिल्लक (पथक पाठवण्यापर्यंत)', breach:'मुदत चुकली', met:'मुदत पूर्ण', view:'पहा', adv:'पुढे न्या', noalert:'कोणताही खुला अलर्ट नाही.',
  compl:'डेटा पूर्णता', gapwarn:'ताजा पशु अहवाल उपलब्ध नाही — गुण मागील मूल्यांवर आधारित', a_alert:'स्पिलओव्हर रडार सूचना', a_risk:'जोखीम', a_animal:'पशू', a_deaths:'मृत्यू', a_abort:'गर्भपात', a_human:'मानवी तापाचे रुग्ण', a_link:'संबंधित जिल्हे',
  a_action:'कृती', a_c:'48 तासांत पशुवैद्यकीय पथक पाठवा; जिल्हा आरोग्य अधिकाऱ्यांना कळवा.', a_w:'अहवालांची खात्री करा; निगराणी वाढवा.', a_n:'नियमित निगराणी.', a_foot:'केवळ निर्णय-सहाय्य; प्रयोगशाळा/क्षेत्र तपासणीने खात्री करा.',
  'Livestock Deaths, Abortions & PPR': 'पशु मृत्यू, गर्भपात आणि पीपीआर', 'Thermal Shock & Rainfall Deficit': 'थर्मल शॉक आणि पावसाची कमतरता', 'Vaccination Coverage Deficit': 'लसीकरण कव्हरेजची कमतरता'
 } 
};
Object.assign(I18N.en, { ban:'critical district(s) awaiting dispatch (48 h target)', ban_open:'open Alert Center', ac_title:'Alert Center', ac_sim:'Simulate field-report surge', ac_notif:'Enable desktop notifications', ac_copy:'Copy message', ac_map:'Show on map',
  ac_help:'An alert opens automatically when a district reaches CRITICAL (score \u226570). Target: dispatch a veterinary team within 48 h. Advance each alert through the workflow; the SLA clock runs from the time it opened.', wh_on:'Webhook notifications: ON', wh_off:'Webhook: off (start run.py with --webhook URL to push alerts to Slack/WhatsApp/SMS gateways)' });
Object.assign(I18N.hi, { ban:'गंभीर जिले टीम भेजने की प्रतीक्षा में (48 घंटे लक्ष्य)', ban_open:'अलर्ट केंद्र खोलें', ac_title:'अलर्ट केंद्र', ac_sim:'क्षेत्र रिपोर्ट में उछाल का अनुकरण करें', ac_notif:'डेस्कटॉप सूचनाएँ चालू करें', ac_copy:'संदेश कॉपी करें', ac_map:'मानचित्र पर देखें',
  ac_help:'कोई जिला गंभीर (स्कोर ≥70) होते ही अलर्ट अपने-आप खुल जाता है। लक्ष्य: 48 घंटे में पशु चिकित्सा टीम भेजना। हर अलर्ट को चरण-दर-चरण आगे बढ़ाएँ; समय-सीमा अलर्ट खुलने से गिनी जाती है।', wh_on:'वेबहुक सूचनाएँ: चालू', wh_off:'वेबहुक: बंद (अलर्ट Slack/WhatsApp/SMS पर भेजने के लिए run.py --webhook URL से चलाएँ)' });
Object.assign(I18N.mr, { ban:'गंभीर जिल्ह्यांत पथक पाठवणे बाकी (48 तासांचे लक्ष्य)', ban_open:'अलर्ट केंद्र उघडा', ac_title:'अलर्ट केंद्र', ac_sim:'क्षेत्र अहवालातील वाढ सिम्युलेट करा', ac_notif:'डेस्कटॉप सूचना चालू करा', ac_copy:'संदेश कॉपी करा', ac_map:'नकाशावर पहा',
  ac_help:'एखादा जिल्हा गंभीर (गुण ≥70) होताच अलर्ट आपोआप उघडतो. लक्ष्य: 48 तासांत पशुवैद्यकीय पथक पाठवणे. प्रत्येक अलर्ट टप्प्याटप्प्याने पुढे न्या; मुदत अलर्ट उघडल्यापासून मोजली जाते.', wh_on:'वेबहुक सूचना: चालू', wh_off:'वेबहुक: बंद (अलर्ट Slack/WhatsApp/SMS वर पाठवण्यासाठी run.py --webhook URL वापरा)' });

let lang = 'en';
const seen = new Set(); let firstLoad = true;
const T = k => (I18N[lang] && I18N[lang][k]) || I18N.en[k] || k;
const lbl = st => T(st.toLowerCase());

const distHi = {"Barmer":"बाड़मेर", "Jodhpur":"जोधपुर", "Bikaner":"बीकानेर", "Jaisalmer":"जैसलमेर", "Jaipur":"जयपुर", "Kutch":"कच्छ", "Ahmednagar":"अहमदनगर", "Pune":"पुणे", "Latur":"लातूर", "Bhopal":"भोपाल", "Wayanad":"वायनाड", "Kozhikode":"कोझिकोड"};
const distMr = {"Barmer":"बाडमेर", "Jodhpur":"जोधपूर", "Bikaner":"बिकानेर", "Jaisalmer":"जैसलमेर", "Jaipur":"जयपूर", "Kutch":"कच्छ", "Ahmednagar":"अहमदनगर", "Pune":"पुणे", "Latur":"लातूर", "Bhopal":"भोपाळ", "Wayanad":"वायनाड", "Kozhikode":"कोझिकोड"};
const trDist = nm => lang === 'hi' ? (distHi[nm] || nm) : lang === 'mr' ? (distMr[nm] || nm) : nm;

function applyLang() { 
  document.querySelectorAll('[data-i]').forEach(e => {
    const key = e.dataset.i;
    if (T(key)) e.textContent = T(key);
  });
}
let P = __DATA__, D, byId, W = P.weights.slice(), REF = P.ref_cap, LAST, wk, selected = null, playing = null, vaxPlus = 0, inited = false;
let activeHighlightWeek = null; 
const markers = {}, buffers = {}, corridors = {}, labels = {}; let heat = null;
const grp = {};
const colorFor = s => s >= 70 ? '#ef4444' : s >= 40 ? '#f59e0b' : '#10b981';
const statusFor = s => s >= 70 ? 'CRITICAL' : s >= 40 ? 'WATCH' : 'NORMAL';
const actionFor = st => T('act_' + st.toLowerCase());
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function toast(m, bad) { const t = $('toast'); if(!t) return; t.textContent = m; t.style.background = bad ? '#ef4444' : '#f472b6'; t.style.display = 'block'; setTimeout(() => t.style.display = 'none', 3200); }

const score = (d, w, vc) => {
  const v = Math.min(1, d.vet[w] / REF[0]), c = Math.min(1, d.clim[w] / REF[1]),
        x = Math.min(1, (1 - (vc == null ? d.vax_cov : vc)) * 10 / REF[2]);
  return Math.round((v * W[0] + c * W[1] + x * W[2]) * 1000) / 10; };
const shareOf = (d, w) => { const a = [Math.min(1, d.vet[w] / REF[0]) * W[0], Math.min(1, d.clim[w] / REF[1]) * W[1], Math.min(1, (1 - d.vax_cov) * 10 / REF[2]) * W[2]];
  const t = a.reduce((x, y) => x + y, 0) || 1; return a.map(v => Math.round(v / t * 100)); };
function recompute() { D.forEach(d => d.S = d.vet.map((_, i) => score(d, i))); }
function trend(d) { const k = Math.min(4, d.S.length), y = d.S.slice(-k), xm = (k - 1) / 2, ym = y.reduce((a, b) => a + b, 0) / k;
  let nu = 0, de = 0; y.forEach((v, i) => { nu += (i - xm) * (v - ym); de += (i - xm) ** 2; });
  const m = de ? nu / de : 0, last = y[k - 1];
  return { m, pts: [1, 2].map(h => Math.max(0, Math.min(100, last + m * h))), toCrit: (last < 70 && m > 0.5) ? Math.ceil((70 - last) / m) : null }; }

/* --- SEAMLESS EXCEL / CSV HANDLING VIA SHEETJS --- */
async function handleFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;
  
  const reader = new FileReader();
  reader.onload = async function(e) {
    try {
      const data = new Uint8Array(e.target.result);
      const workbook = XLSX.read(data, {type: 'array'});
      const firstSheet = workbook.SheetNames[0];
      const csvText = XLSX.utils.sheet_to_csv(workbook.Sheets[firstSheet]);
      
      const p = await post('/api/upload', { csv_data: csvText });
      ingest(p); 
      wk = LAST; 
      if($('week')) $('week').value = LAST; 
      render(); 
      toast('Data Uploaded & Dashboard Updated');
    } catch(err) {
      toast('Upload failed: ' + err.message, true);
    }
  };
  reader.readAsArrayBuffer(file);
  event.target.value = '';
}

/* --- GOOGLE SHEETS DIRECT SYNC --- */
async function importGoogleSheet() {
  const url = prompt("Paste your Public Google Sheet URL here:\n(Make sure 'Anyone with the link can view' is enabled)");
  if (!url) return;
  
  toast('Fetching Google Sheet...');
  try {
    let fetchUrl = url;
    if (url.includes('/edit')) {
      fetchUrl = url.replace(/\/edit.*$/, '/export?format=csv');
    }
    const p = await post('/api/fetch_url', { url: fetchUrl });
    ingest(p);
    wk = LAST; 
    if($('week')) $('week').value = LAST; 
    render(); 
    toast('Google Sheet Synced & Dashboard Updated');
  } catch (err) {
    toast('Failed to sync: ' + err.message, true);
  }
}

function ingest(p) {
  P = p; D = p.districts; byId = Object.fromEntries(D.map(d => [d.id, d])); LAST = p.weeks.length - 1;
  if (wk === undefined || wk > LAST) wk = LAST;
  if($('week')) $('week').max = LAST; if (!inited && $('week')) $('week').value = LAST;
  recompute();
  if (!inited) initMap();
  checkNew();
  inited = true;
}

function initMap() {
  window.map = L.map('map', { minZoom: 4 }).setView([22.5937, 78.9629], 5);
  const dark = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}', { attribution: 'Tiles &copy; Esri', maxZoom: 16 });
  const street = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', { attribution: '&copy; OpenStreetMap contributors', maxZoom: 18 });
  dark.addTo(map);
  let tErr = 0;
  dark.on('tileerror', () => { if (++tErr === 4 && map.hasLayer(dark)) { map.removeLayer(dark); street.addTo(map); } });
  L.control.layers({ 'Dark (Esri)': dark, 'Street (OSM)': street }, null, { position: 'topright' }).addTo(map);
  
  grp.markers = L.layerGroup(); grp.buf = L.layerGroup(); grp.cor = L.layerGroup(); grp.labels = L.layerGroup();
  L.control.scale({ metric: true, imperial: false, position: 'bottomright' }).addTo(map);
  const lg = L.control({ position: 'bottomleft' });
  lg.onAdd = () => { const el = L.DomUtil.create('div', 'maplegend'); el.innerHTML = '<b>Risk</b><br><i class="dot" style="background:#ef4444"></i>Critical \u226570<br><i class="dot" style="background:#f59e0b"></i>Watch 40\u201369<br><i class="dot" style="background:#10b981"></i>Normal &lt;40'; return el; };
  lg.addTo(map);
  const hc = L.control({ position: 'topleft' });
  hc.onAdd = () => { const el = L.DomUtil.create('div', 'leaflet-bar'); el.innerHTML = '<a href="#" title="Fit all districts" id="fitAll">\u25A3</a><a href="#" title="Zoom to critical districts" id="fitCrit">\u26A0</a>'; L.DomEvent.disableClickPropagation(el); return el; };
  hc.addTo(map);
  if($('fitAll')) $('fitAll').onclick = e => { e.preventDefault(); fitAll(); };
  if($('fitCrit')) $('fitCrit').onclick = e => { e.preventDefault(); const c = D.filter(d => d.S[wk] >= 70); if (c.length) map.fitBounds(L.latLngBounds(c.map(d => [d.lat, d.lon])).pad(0.8), { maxZoom: 8 }); };
  const chip = L.control({ position: 'topright' });
  chip.onAdd = () => { const el = L.DomUtil.create('div', 'mapchip'); el.id = 'mapchip'; return el; };
  chip.addTo(map);
  
  D.forEach(d => {
    markers[d.id] = L.marker([d.lat, d.lon], { icon: mkIcon(d), riseOnHover: true }).on('click', () => select(d.id));
    buffers[d.id] = L.circle([d.lat, d.lon], { radius: 5000, weight: 1, fillOpacity: .25 });
    labels[d.id] = L.marker([d.lat, d.lon], { icon: L.divIcon({ className: '', iconSize: [0,0], html: '<div class="mkl" style="position:absolute; transform:translate(-50%, 14px); color:#f5f3ff; text-shadow:0 0 4px #000; font-weight:bold; pointer-events:none; font-size:11px;">' + esc(trDist(d.name)) + '</div>' }) });
    
    // Add layers to groups, but DO NOT add the groups to the map yet!
    grp.markers.addLayer(markers[d.id]);
    grp.buf.addLayer(buffers[d.id]);
    grp.labels.addLayer(labels[d.id]);
  });
  
  P.corridors.forEach(c => { const a = byId[c.from], b = byId[c.to];
    if(a && b) {
      const line = L.polyline(curve(a, b), { weight: 1.5 + c.vol * 3, dashArray: '8 6', className: 'flow', opacity: .85 });
      grp.cor.addLayer(line);
      corridors[c.name] = { c, line };
    }
  });
  if($('rDist')) $('rDist').innerHTML = D.map(d => '<option value="' + esc(d.name) + '">' + esc(d.name) + ', ' + esc(d.state) + '</option>').join('');
  
  // NOTE: Markers and labels are NOT added to the map initially.

  const tg = (id, key) => { if($(id)) $(id).addEventListener('change', e => { if (e.target.checked) grp[key].addTo(map); else map.removeLayer(grp[key]); updateHint(); render(); }); };
  tg('lMarkers', 'markers'); tg('lBuf', 'buf'); tg('lCor', 'cor'); tg('lLab', 'labels');
  
  if($('lHeat')) $('lHeat').addEventListener('change', e => { updateHint(); if (!heat) return; e.target.checked ? heat.addTo(map) : map.removeLayer(heat); });
  fetch('data/india_boundary.geojson').then(r => r.ok ? r.json() : null).then(g => { if (g) L.geoJSON(g, { style: { color: '#c084fc', weight: 1.5, fill: false }, interactive: false }).addTo(map); }).catch(() => {});
  setTimeout(() => { map.invalidateSize(); fitAll(); }, 250);
}

function fitAll() { try { map.fitBounds(L.latLngBounds(D.map(d => [d.lat, d.lon])).pad(0.35)); } catch (e) {} }
function mkIcon(d) { const s = d.S[wk], sz = Math.round(22 + s / 5), c = colorFor(s);
  return L.divIcon({ className: '', iconSize: [sz, sz], iconAnchor: [sz / 2, sz / 2],
    html: '<div class="mkr' + (s >= 70 ? ' crit' : '') + (selected === d.id ? ' sel' : '') + '" style="--c:' + c + ';--s:' + sz + 'px">' + Math.round(s) + '</div>' }); }
function curve(a, b) { const mx = (a.lat + b.lat) / 2, my = (a.lon + b.lon) / 2, dx = b.lat - a.lat, dy = b.lon - a.lon, k = .18, cx = mx - dy * k, cy = my + dx * k, out = [];
  for (let i = 0; i <= 16; i++) { const t = i / 16; out.push([(1 - t) * (1 - t) * a.lat + 2 * (1 - t) * t * cx + t * t * b.lat, (1 - t) * (1 - t) * a.lon + 2 * (1 - t) * t * cy + t * t * b.lon]); }
  return out; }
function tip(d) { const s = d.S[wk], top = shareOf(d, wk).map((v, i) => [v, i]).sort((a, b) => b[0] - a[0])[0];
  return '<b>' + esc(trDist(d.name)) + '</b>, ' + esc(d.state) + '<br><span style="color:' + colorFor(s) + '"><b>' + s + '</b> ' + lbl(statusFor(s)) + '</span> \u00B7 ' + esc(T(P.factorNames[top[1]]) || P.factorNames[top[1]]) + ' ' + top[0] + '%<br>' + T('a_deaths') + ': ' + d.deaths[wk] + ' \u00B7 ' + T('a_abort') + ': ' + d.abortions[wk] + ' \u00B7 ' + T('k_hf') + ': ' + d.human[wk] + (d.gap ? '<br>\u26A0 ' + esc(T('gapwarn')) : ''); }
function exposureMap() { const e = {};
  P.corridors.forEach(c => {
    const o = byId[c.from], t = byId[c.to];
    if(o && t) {
      [[c.from, c.to], [c.to, c.from]].forEach(([orig, dest]) => {
        const O = byId[orig], T = byId[dest];
        if(O && T) {
          e[dest] = Math.min(100, (e[dest] || 0) + O.S[wk] / 100 * c.vol * (1 - T.vax_cov) * 100 * 1.6);
        }
      });
    }
  });
  return e; }

function render() {
  if($('weekLbl')) $('weekLbl').innerText = P.weeks[wk]; 
  if($('rankWk')) $('rankWk').innerText = P.weeks[wk];
  let crit = 0, watch = 0, an = 0, hu = 0; const pts = [];
  D.forEach(d => { const s = d.S[wk], col = colorFor(s);
    if(markers[d.id]) {
      markers[d.id].setIcon(mkIcon(d)); 
      markers[d.id].unbindTooltip().bindTooltip(tip(d), { direction: 'top', offset: [0, -14], opacity: .97, className: 'mtip' });
    }
    if(labels[d.id]) {
      labels[d.id].setIcon(L.divIcon({ className: '', iconSize: [0,0], html: '<div class="mkl" style="position:absolute; transform:translate(-50%, 14px); color:#f5f3ff; text-shadow:0 0 4px #000; font-weight:bold; pointer-events:none; font-size:11px;">' + esc(trDist(d.name)) + '</div>' }));
    }
    if(buffers[d.id]) buffers[d.id].setStyle({ color: col, fillColor: col });
    if (s >= 70) crit++; else if (s >= 40) watch++; an += d.deaths[wk]; hu += d.human[wk]; pts.push([d.lat, d.lon, Math.max(.05, s / 100)]); });
  if ($('mapchip')) $('mapchip').innerHTML = '\u{1F6A8} ' + crit + ' ' + lbl('crit').toLowerCase() + ' \u00B7 ' + watch + ' ' + lbl('watch').toLowerCase() + ' \u00B7 ' + P.weeks[wk];
  if($('kCrit')) $('kCrit').innerText = crit; 
  if($('kWatch')) $('kWatch').innerText = watch; 
  if($('kAnimal')) $('kAnimal').innerText = an; 
  if($('kHuman')) $('kHuman').innerText = hu;
  if (L.heatLayer) { if (heat) map.removeLayer(heat);
    heat = L.heatLayer(pts, { radius: 55, blur: 40, minOpacity: .3, gradient: { .3: '#10b981', .55: '#c084fc', .8: '#ec4899' } });
    if ($('lHeat') && $('lHeat').checked) heat.addTo(map); }
  const ex = exposureMap();
  Object.values(corridors).forEach(({ c, line }) => { 
    const a = byId[c.from], b = byId[c.to];
    if(a && b) {
      const hot = Math.max(a.S[wk], b.S[wk]), sl = selected && (c.from === selected || c.to === selected);
      line.setStyle({ color: colorFor(hot), opacity: selected ? (sl ? 1 : .35) : .85, weight: 1.5 + c.vol * 3 + (sl ? 2 : 0) });
      line.unbindTooltip().bindTooltip(c.name + ': ' + trDist(a.name) + ' \u2194 ' + trDist(b.name) + ' (vol ' + c.vol + ')', { className: 'mtip', sticky: true }); 
    }
  });
  const eRows = Object.entries(ex).filter(([, v]) => v >= 5).sort((a, b) => b[1] - a[1]).slice(0, 5);
  if($('exposure')) $('exposure').innerHTML = eRows.length ? eRows.map(([id, v]) => '<div>' + esc(byId[id] ? trDist(byId[id].name) : id) + ' <b style="color:' + colorFor(v) + '">' + Math.round(v) + '</b>' +
    (byId[id] && byId[id].S[wk] >= 70 ? ' <span class="pill" style="background:#ef4444">already critical</span>' : v >= 40 ? ' <span class="pill" style="background:#f59e0b">pre-position team</span>' : '') +
    '</div><div class="bar"><i style="width:' + v + '%;background:' + colorFor(v) + '"></i></div>').join('') : 'No meaningful downstream exposure this week.';
  if($('rank')) $('rank').innerHTML = '<tr><th data-i="rank_dist">District</th><th data-i="rank_score">Score</th><th data-i="rank_status">Status</th><th>ML</th></tr>' + D.slice().sort((x, y) => y.S[wk] - x.S[wk]).map(d => {
    const s = d.S[wk], dl = s - d.S[Math.max(0, wk - 2)], ar = dl >= 8 ? ' <span style="color:#ef4444">&#9650;</span>' : '';
    return '<tr class="row' + (selected === d.id ? ' sel' : '') + '" onclick="select(\'' + d.id + '\')"><td>' + esc(trDist(d.name)) + ar + (d.gap ? ' <span title="' + esc(T('gapwarn')) + '" style="color:#f59e0b">\u26A0</span>' : '') + '</td><td><b style="color:' + colorFor(s) + '">' + s +
      '</b></td><td><span class="pill" style="background:' + colorFor(s) + '">' + lbl(statusFor(s)) + '</span></td><td>' + d.ml_score[wk] + '</td></tr>'; }).join('');
  if (!selected || !byId[selected]) selected = D[0].id;
  showDistrict(byId[selected]); renderAlerts(); renderMeta();
  applyLang();
}

function select(id) {
  selected = id; vaxPlus = 0; if($('vx')) $('vx').value = 0;
  if ($('lMarkers') && !$('lMarkers').checked) { $('lMarkers').checked = true; grp.markers.addTo(map); updateHint(); }
  render();
  const d = byId[id]; if(d && map) map.panTo([d.lat, d.lon]);
}
function zoomBuf() { if($('lBuf') && buffers[selected]) { $('lBuf').checked = true; grp.buf.addTo(map); updateHint(); map.fitBounds(buffers[selected].getBounds(), { maxZoom: 12 }); } }
function dispatchDVO() { toast('WhatsApp Alert Dispatched to District Veterinary Officer!'); }

function showDistrict(d) {
  if(!d) return;
  const s = d.S[wk], st = statusFor(s), tr = trend(d);
  if($('dName')) $('dName').innerText = trDist(d.name) + ', ' + d.state;
  
  const outBadge = $('dOutlook');
  if (outBadge) {
    if (tr.m > 0.5) { outBadge.innerText = '📈 Worsening (14-day)'; outBadge.style.background = '#ef4444'; }
    else if (tr.m < -0.5) { outBadge.innerText = '📉 Improving (14-day)'; outBadge.style.background = '#10b981'; }
    else { outBadge.innerText = '➡️ Stabilizing (14-day)'; outBadge.style.background = '#f59e0b'; }
  }

  if($('dTeams')) $('dTeams').innerText = d.teams;
  if($('dVax')) $('dVax').innerText = (d.vaccines).toLocaleString();

  if($('dStatus')) $('dStatus').innerText = T('status') + ': ' + lbl(st) + ' | ' + T('a_deaths') + ': ' + d.deaths[wk] + ', ' + T('a_abort') + ': ' + d.abortions[wk] + ', PPR/FMD: ' + d.ppr[wk];
  if($('dScore')) { $('dScore').innerText = s + ' / 100'; $('dScore').style.color = colorFor(s); }
  const si = score(d, wk, Math.min(1, d.vax_cov + vaxPlus / 100));
  if($('vxv')) $('vxv').innerText = vaxPlus; 
  if($('dWhat')) $('dWhat').innerHTML = vaxPlus ? 'with vaccination: <b style="color:' + colorFor(si) + '">' + si + '</b> (' + (si - s).toFixed(1) + ')' : '';
  if($('dMeta')) $('dMeta').innerText = T('k_hf') + ': ' + d.human[wk] + ' | Vax coverage: ' + Math.round(d.vax_cov * 100) + '% | Temp spike: ' + d.temp[wk] + '\u00B0C | Rain deficit: ' + d.rain[wk] + '%';
  
  if($('dFactorsComb')) {
    const shares = shareOf(d, wk);
    $('dFactorsComb').innerHTML = shares.map((v, i) => {
      const ml = d.ml_shares[i];
      return `<div class="sm" style="display:flex; justify-content:space-between; margin-top:4px;">
                <span>${T(P.factorNames[i]) || P.factorNames[i]}</span> <span style="color:#fbcfe8;font-size:10px;">Wt: ${v}% | AI: ${ml}%</span>
              </div>
              <div class="bar" style="display:flex; background:#191e42;">
                <i style="width:${v}%; background:#f472b6;"></i><i style="width:${ml}%; background:#c084fc; opacity:0.8;"></i>
              </div>`;
    }).join('');
  }

  if($('dDet')) $('dDet').innerHTML = 'Isolation Forest: <b>' + d.ml_score[wk] + '</b>/100' + (d.ml_anomaly ? ' <span class="pill" style="background:#ef4444">ANOMALY</span>' : '') +
    ' &middot; CUSUM change-point: <b>' + (d.cusum_week == null ? 'none' : P.weeks[d.cusum_week]) + '</b>';
  if($('dAction')) $('dAction').innerText = actionFor(st);
  if($('dGap')) $('dGap').innerText = T('compl') + ': ' + d.complete + '%' + ((d.gap || d.complete < 70) ? ' \u26A0 ' + T('gapwarn') : '');
  if($('dTrend')) $('dTrend').innerHTML = 'Trend ' + (tr.m >= 0 ? '+' : '') + tr.m.toFixed(1) + '/wk \u00B7 projected ' + tr.pts.map(v => Math.round(v)).join(' \u2192 ') +
    (tr.toCrit ? ' \u00B7 <b style="color:#f59e0b">could reach CRITICAL in ~' + tr.toCrit + ' wk</b>' : '') + ' <span style="color:#a5b4fc">(linear extrapolation)</span>';
  
  const Wd = 380, H = 96, n = d.S.length, hmax = Math.max(1, ...d.human);
  const x = i => 10 + i * (Wd - 20) / Math.max(1, n - 1), ys = v => H - 8 - v / 100 * (H - 16), yh = v => H - 8 - v / hmax * (H - 16);
  const l1 = d.S.map((v, i) => x(i) + ',' + ys(v)).join(' ');
  const l2 = d.human.map((v, i) => x(i) + ',' + yh(v)).join(' ');
  
  let highlightHTML = '';
  if (activeHighlightWeek !== null && activeHighlightWeek >= 0 && activeHighlightWeek < n) {
    const hx = x(activeHighlightWeek);
    highlightHTML = `<rect x="${hx - 12}" y="4" width="24" height="${H-8}" fill="rgba(244, 114, 182, 0.25)" stroke="#f472b6" stroke-dasharray="2 2" rx="4"/>`;
  }

  const sparkEl = $('spark');
  if(sparkEl) {
    sparkEl.innerHTML = highlightHTML +
      '<line x1="10" x2="370" y1="' + ys(70) + '" y2="' + ys(70) + '" stroke="#ef4444" stroke-dasharray="3 4" opacity=".5"/>' +
      '<polyline fill="none" stroke="#f472b6" stroke-width="2.5" points="' + l1 + '"/>' +
      '<polyline fill="none" stroke="#c084fc" stroke-width="1.5" stroke-dasharray="4 3" points="' + l2 + '"/>' +
      d.S.map((v, i) => `<circle cx="${x(i)}" cy="${ys(v)}" r="${i === wk ? 5 : 3.5}" fill="${i === wk ? '#fff' : colorFor(v)}" stroke="#f472b6" stroke-width="1.5" style="cursor:pointer;" data-week="${i}"/>`).join('') +
      '<circle cx="' + x(wk) + '" cy="' + ys(d.S[wk]) + '" r="6" fill="' + colorFor(s) + '" stroke="#fff" stroke-width="2"/>';
  }
}

const sparkSvg = $('spark');
let isDraggingGraph = false;

function getWeekFromEvent(e) {
  if(!sparkSvg) return 0;
  const rect = sparkSvg.getBoundingClientRect();
  const clickX = e.clientX - rect.left;
  const Wd = rect.width || 380;
  const n = byId[selected].S.length;
  const innerW = Wd - 20;
  const pxPerWk = innerW / Math.max(1, n - 1);
  const rawIdx = Math.round((clickX - (10 * Wd / 380)) / pxPerWk);
  return Math.max(0, Math.min(n - 1, rawIdx));
}

if(sparkSvg) {
  sparkSvg.addEventListener('mousedown', e => {
    isDraggingGraph = true;
    const wIdx = getWeekFromEvent(e);
    activeHighlightWeek = wIdx;
    wk = wIdx;
    if($('week')) $('week').value = wk;
    render();
  });

  window.addEventListener('mousemove', e => {
    if (!isDraggingGraph) return;
    const wIdx = getWeekFromEvent(e);
    activeHighlightWeek = wIdx;
    wk = wIdx;
    if($('week')) $('week').value = wk;
    render();
  });

  window.addEventListener('mouseup', () => {
    isDraggingGraph = false;
  });

  sparkSvg.addEventListener('mousemove', e => {
    const wIdx = getWeekFromEvent(e);
    const d = byId[selected];
    if (!d) return;
    const tipEl = $('graphTooltip');
    if(tipEl) {
      tipEl.style.display = 'block';
      tipEl.style.left = (e.offsetX + 10) + 'px';
      tipEl.style.top = (e.offsetY - 20) + 'px';
      tipEl.textContent = `${P.weeks[wIdx]}: Score ${d.S[wIdx]} | Deaths ${d.deaths[wIdx]} | Fever ${d.human[wIdx]}`;
    }
  });

  sparkSvg.addEventListener('mouseleave', () => {
    if($('graphTooltip')) $('graphTooltip').style.display = 'none';
  });
}

function renderAlerts() {
  const stepsN = P.steps.length;
  if($('alerts')) $('alerts').innerHTML = P.alerts.length ? P.alerts.map(a => {
    const d = byId[a.id], hrs = (Date.now() - Date.parse(a.opened)) / 36e5, sla = a.step < 2
      ? (hrs > 48 ? '<b style="color:#ef4444">' + T('breach') + ' (' + hrs.toFixed(0) + ' h)</b>' : '<b style="color:#f59e0b">' + (48 - hrs).toFixed(1) + ' ' + T('left') + '</b>') : '<span style="color:#10b981">' + T('met') + '</span>';
    return '<div class="q"><b>' + esc(trDist(a.district)) + '</b> \u00B7 ' + (d ? d.S[LAST] : '?') + ' \u00B7 ' + T('st' + a.step) + ' \u00B7 ' + sla +
      '<div class="steps">' + P.steps.map((_, k) => '<i class="' + (k <= a.step ? 'on' : '') + '"></i>').join('') + '</div>' +
      '<button class="btn alt" onclick="select(\'' + a.id + '\')">' + T('view') + '</button> <button class="btn ok" ' + (a.step >= stepsN - 1 ? 'disabled' : '') + ' onclick="ack(\'' + a.id + '\')">' + T('adv') + ' &rsaquo;</button></div>'; }).join('')
    : '<span class="sm">' + T('noalert') + '</span>';
  renderAC();
}

function renderAC() {
  const pending = P.alerts.filter(a => a.step < 2);
  if($('bellN')) $('bellN').textContent = pending.length; 
  if($('bell')) $('bell').classList.toggle('hot', pending.length > 0);
  const bn = $('banner');
  if(bn) {
    if (pending.length) { const w = pending.map(a => byId[a.id]).filter(Boolean).sort((x, y) => y.S[LAST] - x.S[LAST]);
      bn.style.display = 'block'; bn.innerHTML = '\u{1F6A8} ' + pending.length + ' ' + T('ban') + ': ' + w.slice(0, 3).map(d => esc(trDist(d.name)) + ' (' + d.S[LAST] + ')').join(', ') + ' \u2014 <u>' + T('ban_open') + '</u>'; }
    else bn.style.display = 'none';
  }
  if($('acHelp')) $('acHelp').textContent = T('ac_help') + ' ' + (P.webhook ? T('wh_on') : T('wh_off'));
  if($('acList')) $('acList').innerHTML = P.alerts.length ? P.alerts.map(a => { const d = byId[a.id]; if (!d) return '';
    const hrs = (Date.now() - Date.parse(a.opened)) / 36e5, s = d.S[LAST],
      sla = a.step < 2 ? (hrs > 48 ? '<b style="color:#ef4444">' + T('breach') + ' (' + hrs.toFixed(0) + ' h)</b>' : '<b style="color:#f59e0b">' + (48 - hrs).toFixed(1) + ' ' + T('left') + '</b>') : '<span style="color:#10b981">' + T('met') + '</span>',
      drv = shareOf(d, LAST).map((v, i) => [v, i]).sort((x, y) => y[0] - x[0]).slice(0, 2).map(([v, i]) => esc(T(P.factorNames[i])||P.factorNames[i]) + ' ' + v + '%').join(' \u00B7 ');
    return '<div class="q"><div style="display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap"><div><b style="font-size:14px">' + esc(trDist(a.district)) + '</b> <span class="pill" style="background:' + colorFor(s) + '">' + s + '</span> \u00B7 ' + T('st' + a.step) +
      '<div class="sm">' + drv + '</div><div class="sm">' + sla + ' \u00B7 ' + new Date(a.opened).toLocaleString() + '</div></div>' +
      '<div style="display:flex;gap:6px;flex-wrap:wrap;align-items:flex-start"><button class="btn alt" onclick="closeAC();select(\'' + a.id + '\')">' + T('ac_map') + '</button><button class="btn alt" onclick="copyFor(\'' + a.id + '\')">' + T('ac_copy') + '</button>' +
      '<button class="btn ok" ' + (a.step >= P.steps.length - 1 ? 'disabled' : '') + ' onclick="ack(\'' + a.id + '\')">' + T('adv') + ' &rsaquo;</button></div></div>' +
      '<div class="steps">' + P.steps.map((_, k) => '<i class="' + (k <= a.step ? 'on' : '') + '"></i>').join('') + '</div></div>'; }).join('') : '<div class="sm">' + T('noalert') + '</div>';
  if($('acEvents')) $('acEvents').innerHTML = P.events.slice(0, 10).map(e => '<div style="margin:2px 0"><span style="color:#a5b4fc">' + new Date(e.ts).toLocaleTimeString() + '</span> ' + esc(e.text) + '</div>').join('') || '-';
}

function openAC() { if($('ac')) $('ac').style.display = 'flex'; renderAC(); }
function closeAC() { if($('ac')) $('ac').style.display = 'none'; }
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeAC(); });
function askNotif() { try { Notification.requestPermission().then(p => toast(p === 'granted' ? 'Desktop notifications enabled' : 'Notifications blocked', p !== 'granted')); } catch (e) { toast('Notifications not supported here', true); } }
function beep() { try { const c = new (window.AudioContext || window.webkitAudioContext)(), o = c.createOscillator(), g = c.createGain(); o.connect(g); g.connect(c.destination); o.frequency.value = 880; g.gain.value = .08; o.start(); setTimeout(() => { o.stop(); c.close(); }, 350); } catch (e) {} }
function checkNew() { P.alerts.forEach(a => { if (!seen.has(a.id)) { seen.add(a.id); if (!firstLoad) { toast('\u{1F6A8} NEW CRITICAL ALERT: ' + trDist(a.district), true); beep(); } } }); firstLoad = false; }
function copyFor(id) { const t = alertText(byId[id]); (navigator.clipboard ? navigator.clipboard.writeText(t) : Promise.reject()).then(() => toast('Alert message copied'), () => prompt('Copy:', t)); }
async function simulate() { const c = D.filter(d => !P.alerts.some(a => a.id === d.id)).sort((x, y) => y.S[LAST] - x.S[LAST])[0]; if (!c) return toast('Every district already has an alert'); 
  try { ingest(await post('/api/report', { district: c.name, abortions: 25, deaths: 20, ppr: 30 })); wk = LAST; if($('week')) $('week').value = LAST; selected = c.id; render(); } catch (e) { toast('Needs the run.py server: ' + e.message, true); } }

function renderMeta() {
  if($('fresh')) $('fresh').innerText = 'Updated ' + new Date(P.generated_at).toLocaleTimeString();
  if($('chips')) $('chips').innerHTML = P.sources.map(s => '<span class="chip ' + (/^live/.test(s.mode) ? 'live' : /^(CSV|scenario)/.test(s.mode) ? 'csv' : 'sim') + '" title="' + esc(s.feed + ' \u2014 ' + s.detail) + '">' + esc(s.sector) + ': ' + esc(s.mode.split(' (')[0]) + '</span>').join('');
  if($('events')) $('events').innerHTML = P.events.slice(0, 8).map(e => '<div style="margin:2px 0"><span style="color:#a5b4fc">' + new Date(e.ts).toLocaleTimeString() + '</span> ' + esc(e.text) + '</div>').join('') || 'No activity yet.';
}

async function post(u, b) { const r = await fetch(u, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(b) });
  const j = await r.json(); if (!r.ok) throw new Error(j.error || r.status); return j; }

/* --- OPTIMISTIC UI UPDATING FOR INSTANT FIELD REPORTS --- */
async function sendReport() {
  try { 
    const distInput = $('rDist').value;
    const ab = +$('rAb').value || 0;
    const de = +$('rDe').value || 0;
    const pp = +$('rPp').value || 0;
    
    // Optimistic UI Update locally
    const dObj = D.find(x => x.name === distInput);
    if(dObj) {
      dObj.abortions[LAST] += ab;
      dObj.deaths[LAST] += de;
      dObj.ppr[LAST] += pp;
      dObj.vet[LAST] = dObj.abortions[LAST]*2.5 + dObj.deaths[LAST]*4.0 + dObj.ppr[LAST]*1.5;
      recompute();
      render();
    }
    
    // Sync with backend
    const p = await post('/api/report', { district: distInput, abortions: ab, deaths: de, ppr: pp });
    ingest(p); wk = LAST; if($('week')) $('week').value = LAST; 
    if (dObj) selected = dObj.id; 
    render(); 
    toast('Report ingested \u2014 pipeline re-run');
    ['rAb', 'rDe', 'rPp'].forEach(i => { if($(i)) $(i).value = ''; });
  } catch (e) { toast('Needs the run.py server: ' + e.message, true); } 
}

async function ack(id) { try { ingest(await post('/api/ack', { district: id })); render(); } catch (e) { toast(e.message, true); } }
async function refreshNow() { try { ingest(await post('/api/refresh', {})); render(); toast('Data refreshed'); } catch (e) { toast('Refresh needs the run.py server', true); } }

function download(name, text, type) { const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([text], { type: type || 'text/plain' })); a.download = name; a.click(); }
function exportCSV() { const rows = ['district,state,week,risk_score,status,deaths,abortions,human_fever'];
  D.forEach(d => d.S.forEach((s, i) => rows.push([d.name, d.state, P.weeks[i], s, statusFor(s), d.deaths[i], d.abortions[i], d.human[i]].join(',')))); download('spillover_radar_scores.csv', rows.join('\n'), 'text/csv'); }
function alertText(d) { const s = d.S[wk], top = shareOf(d, wk).map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 2).map(([v, i]) => P.factorNames[i] + ' ' + v + '%').join('; ');
  const out = P.corridors.filter(c => c.from === d.id || c.to === d.id).map(c => byId[c.from === d.id ? c.to : c.from].name).join(', ');
  return T('a_alert') + ' [' + lbl(statusFor(s)) + '] ' + trDist(d.name) + ', ' + d.state + ' \u2014 ' + T('a_risk') + ' ' + s + '/100 (' + P.weeks[wk] + '). ' + top + '. ' + T('a_animal') + ': ' + d.deaths[wk] + ' ' + T('a_deaths') + ', ' + d.abortions[wk] + ' ' + T('a_abort') +
    '. ' + T('a_human') + ': ' + d.human[wk] + '. ' + T('a_link') + ': ' + (out || '-') + '. ' + T('a_action') + ': ' + (s >= 70 ? T('a_c') : s >= 40 ? T('a_w') : T('a_n')) + ' ' + T('a_foot'); }
function copyAlert() { const t = alertText(byId[selected]); (navigator.clipboard ? navigator.clipboard.writeText(t) : Promise.reject()).then(() => toast('Alert message copied'), () => prompt('Copy:', t)); }

function sitrep() {
  const crit = D.filter(d => d.S[wk] >= 70).sort((a, b) => b.S[wk] - a.S[wk]), watch = D.filter(d => d.S[wk] >= 40 && d.S[wk] < 70), ex = exposureMap();
  const titleText = lang === 'hi' ? 'स्पिलओवर रडार — स्थिति रिपोर्ट' : lang === 'mr' ? 'स्पिलओव्हर रडार —स्थिती अहवाल' : 'SPILLOVER RADAR — SITUATION REPORT';
  const genText = lang === 'hi' ? 'उत्पन्न किया गया:' : lang === 'mr' ? 'तयार केला:' : 'Generated:';
  const sumText = lang === 'hi' ? 'सारांश:' : lang === 'mr' ? 'सारांश:' : 'SUMMARY:';
  const critText = lang === 'hi' ? 'गंभीर' : lang === 'mr' ? 'गंभीर' : 'CRITICAL';
  const watchText = lang === 'hi' ? 'निगरानी' : lang === 'mr' ? 'निरीक्षण' : 'WATCHLIST';
  const expoText = lang === 'hi' ? 'गलियारा जोखिम सूचकांक:' : lang === 'mr' ? 'मार्गिका जोखीम निर्देशांक:' : 'DOWNSTREAM EXPOSURE (index):';

  const L2 = [titleText, genText + ' ' + new Date().toLocaleString() + ' | ' + P.weeks[wk], '',
    sumText + ' ' + crit.length + ' ' + critText + ', ' + watch.length + ' ' + watchText + ' (' + D.length + ' districts).', ''];
  
  crit.forEach(d => { 
    L2.push(critText + ' — ' + trDist(d.name) + ', ' + d.state + ' (' + d.S[wk] + '/100)',
    '  ' + (lang === 'hi' ? 'मानव बुखार मामले:' : lang === 'mr' ? 'मानवी तापाचे रुग्ण:' : 'Human fever cases:') + ' ' + d.human[wk],
    '  ' + (lang === 'hi' ? 'कार्रवाई:' : lang === 'mr' ? 'कृती:' : 'Action:') + ' ' + actionFor('CRITICAL'), ''); 
  });
  
  L2.push(watchText + ': ' + (watch.map(d => trDist(d.name) + ' (' + d.S[wk] + ')').join(', ') || 'none'), '', expoText + ' ' + Object.entries(ex).filter(([, v]) => v >= 20).sort((a, b) => b[1] - a[1]).map(([id, v]) => (byId[id] ? trDist(byId[id].name) : id) + ' ' + Math.round(v)).join(', '));
  
  download('spillover_sitrep_' + lang + '_' + new Date().toISOString().slice(0, 10) + '.txt', L2.join('\n')); 
}

function togglePlay() { 
  const b = $('playBtn');
  if (playing) { clearInterval(playing); playing = null; if(b) b.innerHTML = '&#9654; ' + T('play'); return; }
  if(b) b.innerHTML = '&#10074;&#10074; Pause'; wk = 0;
  playing = setInterval(() => { if($('week')) $('week').value = wk; render(); if (wk >= LAST) { clearInterval(playing); playing = null; if(b) b.innerHTML = '&#9654; ' + T('play'); return; } wk++; }, 900); 
}

if($('preset')) $('preset').addEventListener('change', e => setW(e.target.value.split(',').map(Number)));
if($('lang')) $('lang').addEventListener('change', e => { lang = e.target.value; applyLang(); render(); });

function setW(ws) { 
  const t = ws.reduce((a, b) => a + b, 0) || 1; W = ws.map(v => v / t); 
  [0, 1, 2].forEach(i => { if($('s' + i)) { $('s' + i).value = Math.round(W[i] * 100); if($('w' + i)) $('w' + i).innerText = Math.round(W[i] * 100) + '%'; } }); 
  recompute(); render(); 
}
[0, 1, 2].forEach(i => { if($('s' + i)) $('s' + i).addEventListener('input', () => setW([0, 1, 2].map(k => +$('s' + k).value))); });

applyLang(); ingest(P); setW(W);
</script>
</body>
</html>
"""


def write_index():
    with open(os.path.join(BASE_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(HTML_TEMPLATE.replace(
            "__DATA__", json.dumps(STATE["payload"])))

# --------------------------------------------------------------------------
# 4. SERVER (static + JSON API) AND TERMINAL REPORT
# --------------------------------------------------------------------------


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=BASE_DIR, **k)

    def log_message(self, fmt, *args):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.split("?")[0] == "/api/districts.json":
            return self._json(STATE["payload"])
        super().do_GET()

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n > 15 * 1024 * 1024:
                raise ValueError("Payload too large")
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._json({"error": "bad request"}, 400)

        route = self.path.split("?")[0]
        try:
            if route == "/api/upload":
                csv_text = body.get("csv_data", "")
                try:
                    new_df = pd.read_csv(io.StringIO(csv_text))
                    miss = [c for c in REQ_COLS if c not in new_df.columns]
                    if miss:
                        return self._json({"error": f"Missing columns: {miss}"}, 400)
                    with LOCK:
                        STATE["base"] = new_df.reset_index(drop=True)
                        STATE["scenario"] = None
                        STATE["climate_mode"] = "User Uploaded Data"
                        log_event(
                            f"New Data uploaded: {new_df['name'].nunique()} districts")
                except Exception as e:
                    return self._json({"error": str(e)}, 400)

            elif route == "/api/fetch_url":
                target_url = body.get("url", "")
                try:
                    req = urllib.request.Request(
                        target_url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, timeout=10) as r:
                        csv_text = r.read().decode('utf-8')
                    new_df = pd.read_csv(io.StringIO(csv_text))
                    miss = [c for c in REQ_COLS if c not in new_df.columns]
                    if miss:
                        return self._json({"error": f"Missing columns: {miss}"}, 400)
                    with LOCK:
                        STATE["base"] = new_df.reset_index(drop=True)
                        STATE["scenario"] = None
                        STATE["climate_mode"] = "Google Sheets Sync"
                        log_event(
                            f"Google Sheet synced: {new_df['name'].nunique()} districts")
                except Exception as e:
                    return self._json({"error": str(e)}, 400)

            elif route == "/api/report":
                name = str(body.get("district", ""))
                vals = {k: int(body.get(k, 0))
                        for k in ("abortions", "deaths", "ppr")}
                if any(v < 0 or v > 500 for v in vals.values()):
                    return self._json({"error": "values must be 0-500"}, 400)
                with LOCK:
                    b = STATE["base"]
                    m = (b["name"] == name) & (b["week"] == b["week"].max())
                    if not m.any():
                        return self._json({"error": "unknown district"}, 404)
                    for k, v in vals.items():
                        b.loc[m, k] = b.loc[m, k].fillna(0) + v
                    log_event(
                        f"Field report: {name} +{vals['abortions']} abortions, +{vals['deaths']} deaths, +{vals['ppr']} PPR/FMD")

            elif route == "/api/ack":
                with LOCK:
                    a = STATE["alerts"].get(str(body.get("district", "")))
                    if not a:
                        return self._json({"error": "no such alert"}, 404)
                    a["step"] = min(a["step"] + 1, len(STEPS) - 1)
                    log_event(f"{a['district']}: {STEPS[a['step']]}")

            elif route == "/api/refresh":
                if STATE["live"]:
                    apply_live_climate()
                else:
                    log_event(
                        "Refresh requested (simulated mode: start with --live for real weather)")
            else:
                return self._json({"error": "not found"}, 404)
        except (ValueError, TypeError):
            return self._json({"error": "invalid values"}, 400)

        self._json(compute_payload())


class Srv(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def terminal_report(p):
    def st(s): return "CRITICAL" if s >= 70 else "WATCH" if s >= 40 else "NORMAL"
    print("\n[D.02] Pipeline + [D.03] anomaly scoring finished.")
    print(f"{'District':<11}{'State':<16}{'Now':>6}{'Δ2wk':>7}  {'Status':<9}{'IF':>4} {'CUSUM':<6}Top driver")
    for d in sorted(p["districts"], key=lambda x: -x["scores"][-1]):
        s = d["scores"][-1]
        cw = "-" if d["cusum_week"] is None else f"Wk{d['cusum_week'] + 1}"
        print(f"{d['name']:<11}{d['state']:<16}{s:>6}{s - d['scores'][-3]:>+7.1f}  {st(s):<9}{d['ml_score'][-1]:>4} {cw:<6}"
              f"{FACTOR_NAMES[int(np.argmax(d['shares'][-1]))]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="fetch real weather from Open-Meteo")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--build-only", action="store_true",
                    help="write files, print report, exit")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--refresh-min", type=int, default=30)
    ap.add_argument(
        "--scenario", help="CSV of a documented outbreak to replay")
    ap.add_argument("--write-scenario-sample", action="store_true")
    ap.add_argument(
        "--webhook", help="URL to POST JSON to when a new CRITICAL alert opens")
    a = ap.parse_args()

    if a.write_scenario_sample:
        p = write_scenario_sample()
        print("Wrote", p, "\nReplay it with: python run.py --scenario", p)
        sys.exit(0)
    WEBHOOK = a.webhook
    if a.scenario and a.live:
        print("[info] --live ignored in scenario mode (historical data)")
    load_base(a.live and not a.scenario, a.scenario)
    payload = compute_payload()
    write_index()
    terminal_report(payload)
    print("[D.01] Wrote index.html and api/districts.json")
    if a.build_only:
        sys.exit(0)

    if STATE["live"]:
        def loop():
            while True:
                time.sleep(a.refresh_min * 60)
                apply_live_climate()
                compute_payload()
        threading.Thread(target=loop, daemon=True).start()

    port = int(os.environ.get("PORT", a.port))
    try:
        with Srv(("0.0.0.0", port), Handler) as httpd:
            url = f"http://localhost:{port}/index.html"
            print(f"Serving at {url}  (Ctrl+C to stop)")
            if not a.no_browser and "PORT" not in os.environ:
                try:
                    webbrowser.open(url)
                except Exception:
                    pass
            httpd.serve_forever()
    except OSError as e:
        print(f"Could not start server on port {a.port}: {e}")
    except KeyboardInterrupt:
        print("\nServer stopped.")
