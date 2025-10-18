import os
import time
import json
import math
import textwrap
from datetime import datetime, timedelta, timezone

import streamlit as st
import pandas as pd
import numpy as np
import requests
import feedparser
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ------------------------------
# Config
# ------------------------------
st.set_page_config(page_title="Commodity News & Insights", layout="wide")

APP_TZ = timezone(timedelta(hours=3))  # Europe/Istanbul (UTC+3)
CACHE_TTL = 15 * 60  # 15 minutes

# Product registry: add/modify freely
PRODUCTS = {
    "Cocoa": {
        "keywords": ["cocoa", "cacao", "ICC0", "NY cocoa", "ICE cocoa"],
        "ticker": "CC=F",  # ICE Cocoa futures (Yahoo)
        "regions": [
            {"name": "Ivory Coast (Daloa)", "lat": 6.877, "lon": -6.450},
            {"name": "Ghana (Kumasi)", "lat": 6.695, "lon": -1.623}
        ],
        "alt_tickers": ["NIB"],  # ETN
    },
    "Coffee": {
        "keywords": ["coffee", "arabica", "robusta", "KC=F", "LRCc2"],
        "ticker": "KC=F",  # ICE Arabica
        "regions": [
            {"name": "Brazil (Minas Gerais)", "lat": -21.20, "lon": -45.00},
            {"name": "Vietnam (Dak Lak)", "lat": 12.66, "lon": 108.04}
        ],
        "alt_tickers": ["JO"],
    },
    "Wheat": {
        "keywords": ["wheat", "CBOT wheat", "ZW=F", "Euronext milling wheat"],
        "ticker": "ZW=F",
        "regions": [
            {"name": "Kansas (US Plains)", "lat": 38.50, "lon": -98.00},
            {"name": "France (Paris Basin)", "lat": 48.38, "lon": 2.50}
        ],
        "alt_tickers": ["WEAT"],
    },
    "Corn / Maize": {
        "keywords": ["corn", "maize", "ZC=F", "CBOT corn"],
        "ticker": "ZC=F",
        "regions": [
            {"name": "Iowa (US Corn Belt)", "lat": 42.00, "lon": -93.00},
            {"name": "Brazil (Mato Grosso)", "lat": -13.00, "lon": -55.00}
        ],
        "alt_tickers": ["CORN"],
    },
    "Sugar": {
        "keywords": ["sugar", "SB=F", "raw sugar", "ICE sugar"],
        "ticker": "SB=F",
        "regions": [
            {"name": "Brazil (São Paulo)", "lat": -22.00, "lon": -47.00},
            {"name": "India (Maharashtra)", "lat": 18.52, "lon": 73.85}
        ],
        "alt_tickers": ["CANE"],
    },
    "Soybeans": {
        "keywords": ["soybeans", "ZL=F", "ZS=F", "CBOT soy"],
        "ticker": "ZS=F",
        "regions": [
            {"name": "US (Illinois)", "lat": 40.00, "lon": -89.00},
            {"name": "Brazil (Paraná)", "lat": -24.00, "lon": -51.00}
        ],
        "alt_tickers": ["SOYB"],
    },
    "Hazelnut": {
        "keywords": ["hazelnut", "fındık", "Giresun", "Ordu", "Ferrero", "TMO"],
        "ticker": None,  # No standard public futures; we’ll show alt links
        "regions": [
            {"name": "Türkiye (Giresun)", "lat": 40.91698, "lon": 38.38741},
            {"name": "Türkiye (Ordu)", "lat": 40.9833, "lon": 37.8833}
        ],
        "alt_tickers": [],  # n/a
    },
}

# Extra RSS sources per topic (feel free to extend)
EXTRA_RSS = {
    "market": [
        "https://www.reuters.com/markets/commodities/rss",  # Reuters commodities
        "https://www.bloomberg.com/feeds/bbiz/commodities.rss",  # Bloomberg commodities (may redirect)
    ],
    "supply_chain": [
        "https://www.theloadstar.com/feed/",
        "https://www.joc.com/rssfeed"
    ],
    "policy": [
        "https://ec.europa.eu/commission/presscorner/home/en/rss",  # EU press
        "https://www.usda.gov/media/press-releases/feed"  # USDA press
    ],
    "climate": [
        "https://www.gdacs.org/rss.aspx?profile=archiverss",  # disasters archive (broad)
    ],
}

# ------------------------------
# Utilities
# ------------------------------

def human_dt(ts):
    try:
        return datetime.fromtimestamp(ts, tz=APP_TZ).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "-"

def google_news_rss(query: str, lang="en", recent_days=30):
    # Uses Google News RSS search; you can tweak time range via when: parameter
    # e.g., q=coffee when:7d
    return f"https://news.google.com/rss/search?q={requests.utils.quote(query)}%20when%3A{recent_days}d&hl={lang}"

def build_queries(product_name: str, keywords: list[str]) -> dict:
    base = " OR ".join([f'"{k}"' if " " in k else k for k in keywords])
    packs = {
        "market": f"({base}) (price OR futures OR market OR export OR import OR demand OR supply OR harvest)",
        "production": f"({base}) (production OR output OR crush OR milling OR yield OR acreage OR planting OR harvest)",
        "climate": f"({base}) (weather OR climate OR frost OR drought OR flood OR heatwave OR storm OR La Niña OR El Niño)",
        "supply_chain": f"({base}) (freight OR shipping OR logistics OR port OR container OR trucking)",
        "policy": f"({base}) (tariff OR quota OR subsidy OR ban OR regulation OR policy OR TMO OR USDA OR EU)",
        "generic": base,
    }
    return packs

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_rss(url: str) -> list[dict]:
    try:
        feed = feedparser.parse(url)
        items = []
        for e in feed.entries:
            published = None
            if hasattr(e, "published_parsed") and e.published_parsed:
                published = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
            elif hasattr(e, "updated_parsed") and e.updated_parsed:
                published = datetime(*e.updated_parsed[:6], tzinfo=timezone.utc)
            items.append({
                "title": getattr(e, "title", ""),
                "link": getattr(e, "link", ""),
                "published": published,
                "source": getattr(e, "source", {}).get("title") if hasattr(e, "source") else feed.feed.get("title", ""),
                "summary": getattr(e, "summary", ""),
            })
        return items
    except Exception as ex:
        return [{"title": f"RSS fetch error: {url}", "link": "", "published": None, "source": "RSS", "summary": str(ex)}]

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_news_pack(queries: dict, lang="en", recent_days=30, extra_by_topic: dict | None = None) -> dict:
    results = {}
    for topic, q in queries.items():
        url = google_news_rss(q, lang=lang, recent_days=recent_days)
        items = fetch_rss(url)
        # Attach extras
        if extra_by_topic and topic in extra_by_topic:
            for ex_url in extra_by_topic[topic]:
                items += fetch_rss(ex_url)
        # Sort by published desc
        items = sorted(items, key=lambda x: x["published"] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        results[topic] = items
    return results

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_prices(yf_symbol: str, days_back=365):
    if not yf_symbol:
        return pd.DataFrame()
    try:
        data = yf.download(yf_symbol, period=f"{days_back}d", auto_adjust=True, progress=False)
        data = data.reset_index().rename(columns={"Date": "date"})
        return data
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def fetch_open_meteo_hourly(lat, lon, start_date: datetime, end_date: datetime):
    # Hourly weather timeline for temperature and precipitation
    # https://open-meteo.com/en/docs
    base = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "rain",
            "surface_pressure",
            "wind_speed_10m",
            "wind_gusts_10m"
        ]),
        "timezone": "auto"
    }
    try:
        r = requests.get(base, params=params, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as ex:
        return {"error": str(ex)}

def weather_dataframe(resp_json):
    if not resp_json or "hourly" not in resp_json:
        return pd.DataFrame()
    h = resp_json["hourly"]
    df = pd.DataFrame(h)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
    return df

def short_card(title, value, subtitle=None):
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.markdown(f"### {value}")
        if subtitle:
            st.caption(subtitle)

def news_table(items: list[dict], max_rows=15):
    if not items:
        st.info("No items found.")
        return
    rows = []
    for it in items[:max_rows]:
        pub = it["published"].astimezone(APP_TZ).strftime("%Y-%m-%d %H:%M") if it["published"] else ""
        rows.append({
            "Published": pub,
            "Title": it["title"],
            "Source": it.get("source", ""),
            "Link": it["link"]
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, use_container_width=True)

def topic_header(label, emoji):
    st.markdown(f"### {emoji} {label}")

def tradingview_widget(symbol: str, height=480):
    # Embed TradingView for an alternative look at price
    if not symbol:
        return
    sym = symbol
    html = f"""
    <div class="tradingview-widget-container">
      <div id="tv-chart"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
        new TradingView.widget({{
          "autosize": true,
          "symbol": "{sym}",
          "interval": "60",
          "timezone": "Etc/UTC",
          "theme": "light",
          "style": "1",
          "locale": "en",
          "hide_top_toolbar": false,
          "allow_symbol_change": false,
          "container_id": "tv-chart"
        }});
      </script>
    </div>
    """
    st.components.v1.html(html, height=height, scrolling=False)

def price_change_stats(df):
    if df.empty:
        return None
    close = df["Close"]
    latest = close.iloc[-1]
    prev = close.iloc[-2] if len(close) > 1 else np.nan
    d1 = (latest - prev) / prev * 100 if prev and not math.isnan(prev) else np.nan
    w = 5
    if len(close) > w:
        wk = (latest - close.iloc[-w]) / close.iloc[-w] * 100
    else:
        wk = np.nan
    m = 21
    if len(close) > m:
        mo = (latest - close.iloc[-m]) / close.iloc[-m] * 100
    else:
        mo = np.nan
    return {"last": latest, "d1_pct": d1, "w_pct": wk, "m_pct": mo}

def analyze_sentiment(texts: list[str]):
    analyzer = SentimentIntensityAnalyzer()
    scores = []
    for t in texts:
        if not t:
            continue
        s = analyzer.polarity_scores(t)
        scores.append(s["compound"])
    if not scores:
        return np.nan, np.nan
    return float(np.mean(scores)), float(np.std(scores))

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.title("Commodity News & Insights")
product_name = st.sidebar.selectbox("Select product", list(PRODUCTS.keys()), index=0)
lang = st.sidebar.selectbox("News language", ["en", "tr", "fr", "de", "es"], index=0)
recent_days = st.sidebar.slider("News window (days)", 3, 60, 21, step=1)
show_tradingview = st.sidebar.checkbox("Show TradingView chart (if available)", value=True)
max_rows_each = st.sidebar.slider("Max headlines per segment", 5, 50, 15, step=5)

# ------------------------------
# Header
# ------------------------------
st.title("📊 Commodity News & Insights")
st.caption(f"Last refresh: {datetime.now(tz=APP_TZ).strftime('%Y-%m-%d %H:%M')} (UTC+3)")

prod = PRODUCTS[product_name]
queries = build_queries(product_name, prod["keywords"])

colA, colB, colC = st.columns([2, 1, 1])
with colA:
    st.markdown(f"## 🔎 {product_name}")
    st.write(
        "Curated, near-real-time headlines + context: prices, production, climate & logistics. "
        "Extend this app by adding more APIs (FAOSTAT/PSD Online, customs, ship tracking, etc.)."
    )
with colB:
    st.write("")
with colC:
    st.write("")

# ------------------------------
# Market Data
# ------------------------------
topic_header("Market Data", "💹")
m1, m2, m3 = st.columns([2, 1, 1])

with m1:
    if prod["ticker"]:
        dfp = fetch_prices(prod["ticker"], days_back=365)
        if dfp.empty:
            st.warning("No price data available from Yahoo Finance.")
        else:
            st.line_chart(dfp.set_index("date")["Close"], use_container_width=True)
            stats = price_change_stats(dfp)
            if stats:
                c1, c2, c3, c4 = st.columns(4)
                short_card("Last", f"{stats['last']:.2f}")
                short_card("1D %", f"{stats['d1_pct']:.2f}%")
                short_card("1W %", f"{stats['w_pct']:.2f}%")
                short_card("1M %", f"{stats['m_pct']:.2f}%")
    else:
        st.info("No standard public futures symbol for this product. Consider embedding a TradingView chart for a related index or a custom data source.")

with m2:
    if prod["alt_tickers"]:
        st.markdown("**Alternative symbols (ETFs/ETNs):**")
        for t in prod["alt_tickers"]:
            st.code(t)
    st.caption("Data source: Yahoo Finance (auto-adjusted close).")

with m3:
    if show_tradingview and prod["ticker"]:
        st.markdown("**TradingView chart**")
        tradingview_symbol = prod["ticker"]  # You can remap if needed
        tradingview_widget(tradingview_symbol, height=420)

st.divider()

# ------------------------------
# Headlines & Sentiment
# ------------------------------
topic_header("Latest Headlines & Sentiment", "🗞️")
news = fetch_news_pack(queries, lang=lang, recent_days=recent_days, extra_by_topic={"market": EXTRA_RSS.get("market", [])})

# Aggregate a generic mix for sentiment gauge
generic_mix = news.get("generic", [])[:40]
texts_for_sent = [(it["title"] or "") + " " + (it["summary"] or "") for it in generic_mix]
mean_sent, std_sent = analyze_sentiment(texts_for_sent)
c1, c2 = st.columns(2)
with c1:
    short_card("Media tone (avg comp.)", f"{mean_sent:.3f}" if not np.isnan(mean_sent) else "—", "VADER sentiment on recent headlines")
with c2:
    short_card("Sentiment dispersion", f"{std_sent:.3f}" if not np.isnan(std_sent) else "—", "Std. dev. of recent headline tone")

st.markdown("**Top market headlines**")
news_table(news.get("market", []), max_rows=max_rows_each)

st.divider()

# ------------------------------
# Production & Trade
# ------------------------------
topic_header("Production & Trade", "🏭")
st.caption("Add FAOSTAT/USDA PSD API calls here for official figures. Below is a headlines slice; wire in your datasets for charts/tables.")

prod_cols = st.columns(1)
with prod_cols[0]:
    news_table(news.get("production", []), max_rows=max_rows_each)

with st.expander("🔌 How to plug official production data (FAOSTAT/PSD)", expanded=False):
    st.markdown(
        """
        **FAOSTAT**: Use the API `https://fenixservices.fao.org/faostat/api/v1/en/QCL` with filters for item (e.g., cocoa beans), area, and element (production/area/yield).  
        **USDA PSD**: Use PSD Online QuickStats/CSV exports, then cache locally.  
        Add a function like `fetch_faostat(item_code, area_code)` and display a multi-year chart.
        """
    )

st.divider()

# ------------------------------
# Climate & Weather
# ------------------------------
topic_header("Climate & Weather", "⛅")
st.caption("Regional snapshots from Open-Meteo (hourly). Extend with drought indices, soil moisture, seasonal outlooks.")

lookback_days = st.slider("Weather lookback days", 7, 90, 30, key="w_lookback")
now = datetime.now(tz=APP_TZ)
start_w = (now - timedelta(days=lookback_days)).replace(tzinfo=None)
end_w = now.replace(tzinfo=None)

for region in prod["regions"]:
    st.markdown(f"**{region['name']}**")
    rjson = fetch_open_meteo_hourly(region["lat"], region["lon"], start_w, end_w)
    if "error" in rjson:
        st.error(f"Weather error: {rjson['error']}")
        continue
    wdf = weather_dataframe(rjson)
    if wdf.empty:
        st.info("No weather data returned.")
        continue

    cA, cB, cC = st.columns(3)
    with cA:
        st.line_chart(wdf.set_index("time")["temperature_2m"], use_container_width=True)
        st.caption("Temperature (°C)")
    with cB:
        st.line_chart(wdf.set_index("time")["precipitation"], use_container_width=True)
        st.caption("Precipitation (mm)")
    with cC:
        if "wind_speed_10m" in wdf.columns:
            st.line_chart(wdf.set_index("time")["wind_speed_10m"], use_container_width=True)
            st.caption("Wind speed (m/s)")

st.divider()

# ------------------------------
# Supply Chain & Logistics
# ------------------------------
topic_header("Supply Chain & Logistics", "🚢")
news_sc = fetch_news_pack({"supply_chain": queries["supply_chain"]}, lang=lang, recent_days=recent_days, extra_by_topic={"supply_chain": EXTRA_RSS.get("supply_chain", [])})
news_table(news_sc.get("supply_chain", []), max_rows=max_rows_each)

st.divider()

# ------------------------------
# Policy & Regulation
# ------------------------------
topic_header("Policy & Regulation", "🏛️")
news_pol = fetch_news_pack({"policy": queries["policy"]}, lang=lang, recent_days=recent_days, extra_by_topic={"policy": EXTRA_RSS.get("policy", [])})
news_table(news_pol.get("policy", []), max_rows=max_rows_each)

st.divider()

# ------------------------------
# Notes & Extensibility
# ------------------------------
with st.expander("🧩 Extend this app", expanded=False):
    st.markdown(
        """
        - **Add official data**: FAOSTAT, USDA PSD, ICE/CBOT settlement via vendor APIs, customs exports (Comext, UN Comtrade).
        - **Ship tracking & freight**: MarineTraffic, Freightos FBX API, port congestion feeds.
        - **Weather/climate**: ERA5/ECMWF seasonal outlooks, drought monitors, NOAA ENSO updates.
        - **Event mapping**: GDELT events, GDACS, ReliefWeb. Show geospatial pins with `st.map` or `pydeck`.
        - **Alerts**: Add Streamlit `st.button` to save alert rules (keywords, regions, price moves) and run on a schedule (e.g., CRON + webhook).
        - **Multi-language**: Switch `lang` for Google News RSS (`hl=` parameter already supported).
        """
    )

st.caption("Built with ❤️ using Streamlit.")
