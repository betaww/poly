"""
Mac Dashboard — Streamlit-based monitoring and control.

Subscribes to VPS telemetry via Redis and displays:
  - Real-time P&L curve
  - Current positions and orders
  - Brain node status
  - Risk manager state
  - Panic stop button

Usage:
  streamlit run user_data/polymarket/dashboard/app.py
"""
import json
import time
import threading
from dataclasses import dataclass, field

import redis

try:
    import streamlit as st
except ImportError:
    st = None  # graceful import — module can be imported without streamlit

import sys
import os

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from user_data.polymarket.config import Config


def create_redis_client(config: Config) -> redis.Redis:
    rc = config.redis
    kwargs = {
        "host": rc.host,
        "port": rc.port,
        "db": rc.db,
        "decode_responses": True,
    }
    if rc.password:
        kwargs["password"] = rc.password
    return redis.Redis(**kwargs)


def send_command(r: redis.Redis, config: Config, command: str, **kwargs):
    """Send a control command to VPS."""
    msg = {"command": command, **kwargs}
    r.publish(config.redis.ch_control, json.dumps(msg))


def run_dashboard():
    """Main Streamlit app."""
    if st is None:
        print("streamlit not installed. Run: pip install streamlit")
        return

    st.set_page_config(
        page_title="Polymarket Trading Dashboard",
        page_icon="📊",
        layout="wide",
    )

    st.title("📊 Polymarket Trading Dashboard")

    config = Config.from_env()

    try:
        r = create_redis_client(config)
        r.ping()
        st.sidebar.success(f"Redis: {config.redis.host}:{config.redis.port}")
    except Exception as e:
        st.sidebar.error(f"Redis: {e}")
        st.stop()

    # --- Control Panel ---
    st.sidebar.header("🎮 Controls")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("⏸️ Pause", use_container_width=True):
            send_command(r, config, "PAUSE")
            st.toast("Paused!")
    with col2:
        if st.button("▶️ Resume", use_container_width=True):
            send_command(r, config, "RESUME")
            st.toast("Resumed!")

    if st.sidebar.button("🚨 PANIC STOP", type="primary", use_container_width=True):
        send_command(r, config, "PANIC_STOP")
        st.error("PANIC STOP sent!")

    # --- Config Tuning ---
    st.sidebar.header("⚙️ Parameters")
    new_spread = st.sidebar.slider("Base Spread", 0.01, 0.10, 0.03, 0.01)
    if st.sidebar.button("Update Spread"):
        send_command(r, config, "UPDATE_CONFIG", spread=new_spread)
        st.toast(f"Spread → {new_spread}")

    # --- Telemetry Display ---
    # Get latest telemetry from Redis
    telemetry_key = "poly:latest_telemetry"
    placeholder = st.empty()

    # Subscribe to telemetry in background
    ps = r.pubsub()
    ps.subscribe(config.redis.ch_telemetry)

    latest = {}

    # Try to get recent message
    for _ in range(10):
        msg = ps.get_message(timeout=0.2)
        if msg and msg["type"] == "message":
            latest = json.loads(msg["data"])
            break

    if latest:
        # Top metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Daily P&L", f"${latest.get('daily_pnl', 0):+.2f}")
        with col2:
            st.metric("Total P&L", f"${latest.get('total_pnl', 0):+.2f}")
        with col3:
            st.metric("Win Rate", f"{latest.get('win_rate', 0):.1%}")
        with col4:
            st.metric("Rounds", latest.get('rounds_traded', 0))
        with col5:
            brain = "🟢" if latest.get("brain_connected") else "🔴"
            st.metric("Brain", brain)

        # Position
        st.subheader("📌 Current Position")
        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            st.metric("Pos Up", f"{latest.get('position_up', 0):.1f} shares")
        with pcol2:
            st.metric("Pos Down", f"{latest.get('position_down', 0):.1f} shares")
        with pcol3:
            st.metric("Strategy", latest.get("strategy", "—"))

        # Risk
        risk = latest.get("risk", {})
        st.subheader("🛡️ Risk Status")
        rcol1, rcol2 = st.columns(2)
        with rcol1:
            paused = risk.get("is_paused", False)
            st.metric("Status", "⏸️ PAUSED" if paused else "▶️ Running")
        with rcol2:
            st.metric("Daily Volume", f"${risk.get('daily_volume', 0):.0f}")

        # Raw JSON
        with st.expander("Raw Telemetry"):
            st.json(latest)
    else:
        st.info("⏳ Waiting for telemetry from VPS node...")
        st.caption("Make sure the VPS runner is active and Redis is reachable.")

    ps.unsubscribe()

    # Auto-refresh
    time.sleep(2)
    st.rerun()


if __name__ == "__main__":
    run_dashboard()
