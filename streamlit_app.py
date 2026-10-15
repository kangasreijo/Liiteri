import os
import uuid
from datetime import date, timedelta
from urllib.parse import urlencode

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("POLAR_CLIENT_ID")
CLIENT_SECRET = os.getenv("POLAR_CLIENT_SECRET")
REDIRECT_URI = os.getenv("POLAR_REDIRECT_URI", "http://localhost:8501/oauth_callback")

AUTH_URL = "https://auth.polar.com/oauth/authorize"
TOKEN_URL = "https://auth.polar.com/oauth/token"
API_BASE = "https://www.polaraccesslink.com/v3"

SCOPES = [
    "accesslink.read_all",
    "activity:read",
    "sleep:read",
    "training_sessions:read",
    "physical_information:read",
]

st.set_page_config(page_title="Polar Load–Sleep–Activity Analyzer", layout="wide")


def get_query_params():
    return st.query_params


def build_authorization_url():
    state = str(uuid.uuid4())
    st.session_state["oauth_state"] = state

    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": " ".join(SCOPES),
        "state": state,
    }
    return f"{AUTH_URL}?{urlencode(params)}"


def exchange_code_for_token(code: str):
    response = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT_URI,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def auth_headers():
    token = st.session_state.get("access_token")
    if not token:
        raise RuntimeError("No Polar access token available.")
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def api_get(endpoint: str, params: dict | None = None):
    response = requests.get(
        f"{API_BASE}{endpoint}",
        headers=auth_headers(),
        params=params or {},
        timeout=30,
    )

    if response.status_code == 204:
        return None
    response.raise_for_status()
    return response.json()


def get_user_id():
    user = api_get("/users")
    if isinstance(user, dict):
        return user.get("user_id") or user.get("id")
    return None


def load_sleep(start_date: date, end_date: date):
    rows = []

    current = start_date
    while current <= end_date:
        try:
            data = api_get(f"/users/sleep/{current.isoformat()}")
        except requests.HTTPError:
            data = None

        if data:
            rows.append(
                {
                    "date": current,
                    "sleep_start": data.get("sleep_start_time"),
                    "sleep_end": data.get("sleep_end_time"),
                    "total_sleep_minutes": data.get("total_sleep_time", 0) / 60
                    if data.get("total_sleep_time") is not None
                    else None,
                    "sleep_score": data.get("sleep_score"),
                    "deep_sleep_minutes": (data.get("deep_sleep_time") or 0) / 60,
                    "rem_sleep_minutes": (data.get("rem_sleep_time") or 0) / 60,
                    "light_sleep_minutes": (data.get("light_sleep_time") or 0) / 60,
                    "interruptions": data.get("sleep_cycles", 0),
                }
            )
        current += timedelta(days=1)

    return pd.DataFrame(rows)


def load_activity(start_date: date, end_date: date):
    rows = []

    current = start_date
    while current <= end_date:
        try:
            data = api_get(f"/users/activities/{current.isoformat()}")
        except requests.HTTPError:
            data = None

        if data:
            rows.append(
                {
                    "date": current,
                    "steps": data.get("steps"),
                    "active_kcal": data.get("active_kcal"),
                    "daily_activity_goal": data.get("activity_goal"),
                    "activity_distance_m": data.get("distance"),
                }
            )
        current += timedelta(days=1)

    return pd.DataFrame(rows)


def load_training_load(start_date: date, end_date: date):
    rows = []

    current = start_date
    while current <= end_date:
        try:
            data = api_get(
                "/users/cardio-load",
                params={
                    "from": current.isoformat(),
                    "to": current.isoformat(),
                },
            )
        except requests.HTTPError:
            data = None

        if data:
            items = data if isinstance(data, list) else data.get("cardio_loads", [])
            for item in items:
                rows.append(
                    {
                        "date": item.get("date"),
                        "training_load": item.get("load"),
                        "acute_load": item.get("acute"),
                        "chronic_load": item.get("chronic"),
                        "ratio_acute_chronic": item.get("ratio"),
                    }
                )

        current += timedelta(days=1)

    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "training_load",
                "acute_load",
                "chronic_load",
                "ratio_acute_chronic",
            ]
        )

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def create_analysis_dataset(sleep_df, activity_df, load_df):
    for df in (sleep_df, activity_df, load_df):
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"]).dt.date

    merged = load_df.merge(activity_df, on="date", how="outer")
    merged = merged.merge(sleep_df, on="date", how="outer")
    merged = merged.sort_values("date").reset_index(drop=True)

    merged["sleep_debt_proxy"] = (
        merged["total_sleep_minutes"].rolling(3, min_periods=1).mean()
        - merged["total_sleep_minutes"]
    )

    merged["high_load_low_sleep"] = (
        (merged["training_load"] > merged["training_load"].rolling(7, min_periods=3).mean())
        & (merged["total_sleep_minutes"] < merged["total_sleep_minutes"].rolling(7, min_periods=3).mean())
    )

    return merged


def main():
    st.title("Polar Training Load, Sleep & Activity Analyzer")
    st.caption(
        "Secure OAuth 2.0 connection to Polar AccessLink. "
        "Your Polar password is never entered or stored in this app."
    )

    if not CLIENT_ID or not CLIENT_SECRET:
        st.error(
            "Missing POLAR_CLIENT_ID or POLAR_CLIENT_SECRET. "
            "Add them to your .env file."
        )
        st.stop()

    params = get_query_params()

    if "code" in params and "state" in params:
        if params.get("state") != st.session_state.get("oauth_state"):
            st.error("OAuth state mismatch. Please restart the connection.")
        else:
            try:
                token_data = exchange_code_for_token(params["code"])
                st.session_state["access_token"] = token_data.get("access_token")
                st.session_state["refresh_token"] = token_data.get("refresh_token")
                st.success("Polar account connected.")
                st.query_params.clear()
            except Exception as exc:
                st.error(f"Could not exchange authorization code: {exc}")

    col_auth, col_status = st.columns([1, 2])

    with col_auth:
        if st.session_state.get("access_token"):
            st.success("Connected to Polar")
            if st.button("Disconnect"):
                for key in ["access_token", "refresh_token", "oauth_state"]:
                    st.session_state.pop(key, None)
                st.rerun()
        else:
            st.link_button(
                "Connect Polar account",
                build_authorization_url(),
                use_container_width=True,
            )

    with col_status:
        st.info(
            "Click **Connect Polar account**, sign in on Polar’s official page, "
            "and approve the requested data permissions."
        )

    if not st.session_state.get("access_token"):
        st.stop()

    st.divider()
    st.header("Select analysis period")

    today = date.today()
    default_start = today - timedelta(days=27)

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start date", default_start)
    with col2:
        end_date = st.date_input("End date", today)

    if start_date > end_date:
        st.error("Start date must be before or equal to end date.")
        st.stop()

    if st.button("Load Polar data", type="primary"):
        with st.spinner("Loading Training Load, Sleep, and Activity data..."):
            try:
                sleep_df = load_sleep(start_date, end_date)
                activity_df = load_activity(start_date, end_date)
                load_df = load_training_load(start_date, end_date)

                analysis_df = create_analysis_dataset(
                    sleep_df, activity_df, load_df
                )

                st.session_state["analysis_df"] = analysis_df
                st.success("Data loaded successfully.")
            except Exception as exc:
                st.error(f"Polar API request failed: {exc}")

    if "analysis_df" not in st.session_state:
        st.stop()

    df = st.session_state["analysis_df"]

    st.divider()
    st.header("Daily combined dataset")

    if df.empty:
        st.warning("No data available for the selected period.")
        st.stop()

    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download combined CSV",
        csv,
        file_name=f"polar_load_sleep_activity_{start_date}_{end_date}.csv",
        mime="text/csv",
    )

    st.divider()
    st.header("Visual overview")

    numeric_cols = [
        "training_load",
        "total_sleep_minutes",
        "sleep_score",
        "steps",
        "active_kcal",
    ]

    available_cols = [c for c in numeric_cols if c in df.columns]
    selected = st.multiselect(
        "Select variables to plot",
        available_cols,
        default=available_cols[:3],
    )

    if selected:
        plot_df = df.copy()
        plot_df["date"] = pd.to_datetime(plot_df["date"])
        plot_df = plot_df.set_index("date")[selected]
        st.line_chart(plot_df)

    st.divider()
    st.header("Interpretation")

    st.markdown(
        """
        - **Training load** reflects the cardiovascular strain associated with training.
        - **Sleep duration and sleep score** provide context for overnight recovery.
        - **Daily activity** captures non-exercise movement and total daily energy expenditure.
        - The `high_load_low_sleep` flag identifies days where training load was above the recent average while sleep was below the recent average.
        - Use these indicators descriptively; they do not by themselves diagnose overtraining or illness.
        """
    )


if __name__ == "__main__":
    main()
