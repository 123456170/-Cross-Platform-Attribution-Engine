import os
import io
import json
import math
import sqlite3
import tempfile
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    cv2 = None
    CV2_AVAILABLE = False


# ============================================================================
# Constants and domain configuration
# ============================================================================

SCHEMA_VERSION = "1.0.0"
SEED = 20261107
DB_PATH = os.path.join(tempfile.gettempdir(), "cross_platform_attribution_engine_demo.db")

PLATFORMS = [
    "Google Search",
    "Meta Feed",
    "TikTok Video",
    "YouTube",
    "Email",
    "Display Retargeting",
    "Affiliate",
    "Direct",
]

ACTIONABLE_PLATFORMS = [p for p in PLATFORMS if p != "Direct"]

EVENT_FIELDS = [
    "event_id",
    "user_pseudo_id",
    "session_id",
    "timestamp_utc",
    "platform",
    "channel",
    "campaign",
    "ad_group",
    "creative_id",
    "device_type",
    "geo",
    "event_type",
    "cost_micros",
    "impression_count",
    "click_count",
    "consent_flag",
    "fraud_flag",
    "data_quality_flag",
    "source_system",
    "ingestion_time",
    "schema_version",
]

CONVERSION_FIELDS = [
    "conversion_id",
    "user_pseudo_id",
    "session_id",
    "conversion_time_utc",
    "order_id",
    "revenue_micros",
    "margin_micros",
    "product_category",
    "new_customer_flag",
    "coupon_code",
    "consent_flag",
    "fraud_flag",
    "source_system",
    "ingestion_time",
    "schema_version",
]

JOURNEY_FIELDS = [
    "journey_id",
    "user_pseudo_id",
    "session_id",
    "channels_json",
    "terminal",
    "conversion_id",
    "conversion_time",
    "revenue_micros",
    "consent_flag",
    "data_quality_flag",
    "source_system",
    "schema_version",
]

MODEL_MAP = {
    "Markov removal effect": "markov",
    "Last click": "last_click",
    "First click": "first_click",
    "Linear": "linear",
    "Time decay": "time_decay",
    "Position based": "position",
}

SOURCE_BY_PLATFORM = {
    "Google Search": "google_ads_api",
    "Meta Feed": "meta_marketing_api",
    "TikTok Video": "tiktok_ads_api",
    "YouTube": "google_ads_api",
    "Email": "email_service_api",
    "Display Retargeting": "display_bidder_api",
    "Affiliate": "affiliate_network_api",
    "Direct": "internal_ets",
}

CPC_USD = {
    "Google Search": 1.20,
    "Meta Feed": 0.65,
    "TikTok Video": 0.40,
    "YouTube": 0.55,
    "Email": 0.12,
    "Display Retargeting": 0.30,
    "Affiliate": 2.50,
    "Direct": 0.00,
}

CPV_USD = {
    "Google Search": 0.00,
    "Meta Feed": 0.045,
    "TikTok Video": 0.035,
    "YouTube": 0.060,
    "Email": 0.020,
    "Display Retargeting": 0.025,
    "Affiliate": 0.00,
    "Direct": 0.00,
}

CLICK_PROB = {
    "Google Search": 0.86,
    "Meta Feed": 0.52,
    "TikTok Video": 0.22,
    "YouTube": 0.34,
    "Email": 0.72,
    "Display Retargeting": 0.43,
    "Affiliate": 0.78,
    "Direct": 0.00,
}


# ============================================================================
# Utilities
# ============================================================================

def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def to_micros(usd: float) -> int:
    return int(round(float(usd) * 1_000_000))


def from_micros(micros: Any) -> float:
    try:
        if micros is None or pd.isna(micros):
            return 0.0
        return float(micros) / 1_000_000.0
    except Exception:
        return 0.0


def fmt_usd(x: Any) -> str:
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return "$0.00"


def safe_div(a: Any, b: Any) -> float:
    try:
        a = float(a)
        b = float(b)
        if b == 0 or math.isnan(b) or math.isinf(b):
            return 0.0
        return a / b
    except Exception:
        return 0.0


def stable_hash(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]


def json_dumps(obj: Any, indent: Optional[int] = None) -> str:
    return json.dumps(obj, default=str, sort_keys=True, indent=indent)


def is_missing(x: Any) -> bool:
    try:
        return x is None or pd.isna(x)
    except Exception:
        return False


def to_bool(x: Any, default: bool = False) -> Tuple[bool, bool]:
    if is_missing(x):
        return default, True
    if isinstance(x, bool):
        return x, False
    if isinstance(x, (int, float, np.integer, np.floating)):
        return bool(x), False
    s = str(x).strip().lower()
    if s in ("true", "1", "yes", "y"):
        return True, False
    if s in ("false", "0", "no", "n"):
        return False, False
    return default, True


def parse_utc(value: Any) -> Optional[pd.Timestamp]:
    if is_missing(value):
        return None
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts
    except Exception:
        return None


def collapse_sequence(seq: List[str]) -> List[str]:
    out: List[str] = []
    for x in seq:
        if not out or out[-1] != x:
            out.append(x)
    return out


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


# ============================================================================
# Synthetic data generation
# ============================================================================

def _generate_dataset_impl(
    seed: int,
    n_journeys: int,
    reference_time: Optional[datetime] = None,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    now_ref = reference_time or utcnow()
    base_time = now_ref - timedelta(days=90)

    truth_raw = {
        "Google Search": 0.26,
        "Meta Feed": 0.20,
        "Email": 0.15,
        "TikTok Video": 0.12,
        "YouTube": 0.10,
        "Display Retargeting": 0.07,
        "Affiliate": 0.05,
        "Direct": 0.05,
    }
    truth_sum = sum(truth_raw.values())
    truth = {k: v / truth_sum for k, v in truth_raw.items()}

    presence_raw = {
        p: max(0.02, truth[p] * float(rng.normal(1.0, 0.12)))
        for p in ACTIONABLE_PLATFORMS
    }
    presence_sum = sum(presence_raw.values())
    presence = {k: v / presence_sum for k, v in presence_raw.items()}
    presence_arr = np.array([presence[p] for p in ACTIONABLE_PLATFORMS], dtype=float)
    presence_arr /= presence_arr.sum()

    events: List[Dict[str, Any]] = []
    conversions: List[Dict[str, Any]] = []
    journeys: List[Dict[str, Any]] = []

    for j in range(int(n_journeys)):
        user_pseudo_id = f"u_{stable_hash(f'user-{seed}-{j}')}"
        session_id = f"s_{stable_hash(f'session-{seed}-{j}')}"
        consent_flag = bool(rng.random() > 0.04)
        fraud_flag = bool(rng.random() < 0.008)

        direct_only = rng.random() < 0.08
        converts = False
        channels: List[str] = []

        if direct_only:
            converts = rng.random() < 0.28
            channels = ["Direct"] if converts else []
        else:
            length = int(rng.poisson(2.2)) + 1
            length = min(max(length, 1), 6)
            raw_channels = rng.choice(
                ACTIONABLE_PLATFORMS,
                size=length,
                p=presence_arr,
            ).tolist()
            channels = collapse_sequence([str(c) for c in raw_channels])

            base_cvr = 0.22
            len_bonus = 0.04 * len(channels)
            qfactor = (
                sum(truth.get(c, 0.0) for c in channels) / max(1, len(channels))
                if channels
                else 0.5
            )
            cvr = base_cvr + len_bonus + (qfactor - 0.15) * 0.5
            cvr = min(max(cvr, 0.05), 0.65)
            converts = bool(rng.random() < cvr)

        conv_time = base_time + timedelta(
            days=float(rng.uniform(0.0, 89.9)),
            hours=float(rng.uniform(0.0, 23.9)),
        )
        anchor_time = conv_time if converts else conv_time + timedelta(days=1)

        conversion_id = f"c_{stable_hash(f'conv-{seed}-{j}')}" if converts else None
        order_id = f"o_{stable_hash(f'order-{seed}-{j}')}" if converts else None

        revenue_micros = 0
        margin_micros = 0
        product_category = ""
        new_customer_flag = bool(rng.random() < 0.35)
        coupon_code = rng.choice([None, "WELCOME10", "FLASH20", "LOYAL5", None, None])

        if converts:
            revenue_usd = float(rng.lognormal(mean=math.log(85.0), sigma=0.55))
            revenue_usd = min(revenue_usd, 5000.0)
            if rng.random() < 0.002:
                revenue_usd = float(rng.uniform(20000.0, 120000.0))
            product_category = str(
                rng.choice(
                    [
                        "App Subscription",
                        "E-commerce",
                        "Lead Gen",
                        "Local Service",
                        "SaaS Trial",
                    ]
                )
            )
            margin_rate = float(rng.uniform(0.25, 0.65))
            revenue_micros = to_micros(revenue_usd)
            margin_micros = to_micros(revenue_usd * margin_rate)

        lags_hours: List[float] = []
        if channels:
            rev_lags: List[float] = []
            cur = abs(float(rng.exponential(7.0))) + 0.5
            for _ in channels:
                rev_lags.append(cur)
                cur += abs(float(rng.exponential(20.0)))
            lags_hours = [min(x, 24.0 * 13.0) for x in rev_lags[::-1]]

        for idx, platform in enumerate(channels):
            if platform == "Direct":
                continue

            event_time = anchor_time - timedelta(hours=float(lags_hours[idx]))
            r = float(rng.random())
            if r < CLICK_PROB.get(platform, 0.4):
                event_type = "click"
            elif platform in ("TikTok Video", "YouTube"):
                event_type = "video_view"
            elif platform in ("Meta Feed", "Display Retargeting"):
                event_type = "view"
            elif platform == "Email":
                event_type = "email_open"
            else:
                event_type = "impression"

            if event_type == "click":
                base_cost = CPC_USD.get(platform, 0.5)
            elif event_type == "video_view":
                base_cost = CPV_USD.get(platform, 0.05)
            elif event_type == "view":
                base_cost = CPV_USD.get(platform, 0.04) * 0.7
            elif event_type == "email_open":
                base_cost = 0.02
            else:
                base_cost = max(0.003, CPC_USD.get(platform, 0.3) * 0.03)

            cost_usd = max(0.005, float(rng.lognormal(math.log(max(base_cost, 0.005)), 0.35)))
            cost_micros = to_micros(cost_usd)

            event_id = f"e_{stable_hash(f'event-{seed}-{j}-{idx}')}"
            row = {
                "event_id": event_id,
                "user_pseudo_id": user_pseudo_id,
                "session_id": session_id,
                "timestamp_utc": event_time.isoformat(),
                "platform": platform,
                "channel": platform,
                "campaign": f"camp_{platform.split()[0].lower()}_{int(rng.integers(1, 6))}",
                "ad_group": f"ag_{int(rng.integers(1, 21))}",
                "creative_id": f"cr_{int(rng.integers(1, 101))}",
                "device_type": str(rng.choice(["mobile", "desktop", "tablet"])),
                "geo": str(rng.choice(["US", "CA", "UK", "DE", "IN", "BR", "JP", "AU"])),
                "event_type": event_type,
                "cost_micros": cost_micros,
                "impression_count": 1 if event_type in ("impression", "view", "video_view") else 0,
                "click_count": 1 if event_type == "click" else 0,
                "consent_flag": consent_flag,
                "fraud_flag": fraud_flag,
                "data_quality_flag": "",
                "source_system": SOURCE_BY_PLATFORM.get(platform, "internal_ets"),
                "ingestion_time": (event_time + timedelta(minutes=float(rng.exponential(30.0)))).isoformat(),
                "schema_version": SCHEMA_VERSION,
            }

            issue_roll = float(rng.random())
            if issue_roll < 0.008:
                row["platform"] = None
            elif issue_roll < 0.014:
                row["cost_micros"] = -100
            elif issue_roll < 0.020:
                row["timestamp_utc"] = (now_ref + timedelta(days=2)).isoformat()
            elif issue_roll < 0.024:
                row["schema_version"] = "0.9.0"

            events.append(row)

            if rng.random() < 0.010:
                dup = row.copy()
                dup["ingestion_time"] = (event_time + timedelta(days=1)).isoformat()
                events.append(dup)

        if converts and rng.random() < 0.12:
            platform = str(rng.choice(ACTIONABLE_PLATFORMS, p=presence_arr))
            event_time = conv_time + timedelta(hours=float(rng.uniform(1.0, 48.0)))
            event_id = f"e_post_{stable_hash(f'post-{seed}-{j}')}"
            events.append(
                {
                    "event_id": event_id,
                    "user_pseudo_id": user_pseudo_id,
                    "session_id": session_id,
                    "timestamp_utc": event_time.isoformat(),
                    "platform": platform,
                    "channel": platform,
                    "campaign": f"camp_post_{int(rng.integers(1, 5))}",
                    "ad_group": f"ag_post_{int(rng.integers(1, 10))}",
                    "creative_id": f"cr_post_{int(rng.integers(1, 20))}",
                    "device_type": "mobile",
                    "geo": "US",
                    "event_type": "click",
                    "cost_micros": to_micros(0.35),
                    "impression_count": 0,
                    "click_count": 1,
                    "consent_flag": consent_flag,
                    "fraud_flag": False,
                    "data_quality_flag": "",
                    "source_system": SOURCE_BY_PLATFORM.get(platform, "internal_ets"),
                    "ingestion_time": (event_time + timedelta(minutes=10)).isoformat(),
                    "schema_version": SCHEMA_VERSION,
                }
            )

        if converts:
            conv_row = {
                "conversion_id": conversion_id,
                "user_pseudo_id": user_pseudo_id,
                "session_id": session_id,
                "conversion_time_utc": conv_time.isoformat(),
                "order_id": order_id,
                "revenue_micros": revenue_micros,
                "margin_micros": margin_micros,
                "product_category": product_category,
                "new_customer_flag": new_customer_flag,
                "coupon_code": coupon_code,
                "consent_flag": consent_flag,
                "fraud_flag": fraud_flag,
                "source_system": "internal_ets",
                "ingestion_time": (conv_time + timedelta(minutes=float(rng.exponential(15.0)))).isoformat(),
                "schema_version": SCHEMA_VERSION,
            }

            if rng.random() < 0.005:
                conv_row["revenue_micros"] = -100
            if rng.random() < 0.004:
                conv_row["conversion_time_utc"] = None

            conversions.append(conv_row)

            if rng.random() < 0.008:
                dup = conv_row.copy()
                dup["ingestion_time"] = (conv_time + timedelta(days=1)).isoformat()
                conversions.append(dup)

        journeys.append(
            {
                "journey_id": f"j_{stable_hash(f'journey-{seed}-{j}')}",
                "user_pseudo_id": user_pseudo_id,
                "session_id": session_id,
                "channels_json": json.dumps(channels),
                "terminal": "conversion" if converts else "null",
                "conversion_id": conversion_id,
                "conversion_time": conv_time.isoformat(),
                "revenue_micros": revenue_micros,
                "consent_flag": consent_flag,
                "data_quality_flag": "",
                "source_system": "synthetic_journey_generator",
                "schema_version": SCHEMA_VERSION,
            }
        )

    events_df = pd.DataFrame(events, columns=EVENT_FIELDS)
    conversions_df = pd.DataFrame(conversions, columns=CONVERSION_FIELDS)
    journeys_df = pd.DataFrame(journeys, columns=JOURNEY_FIELDS)

    return {
        "events": events_df,
        "conversions": conversions_df,
        "journeys": journeys_df,
        "truth": truth,
        "seed": seed,
        "n_journeys": int(n_journeys),
        "generated_at": now_ref.isoformat(),
    }


@st.cache_data(show_spinner="Generating synthetic cross-platform advertising dataset...")
def generate_dataset(seed: int = SEED, n_journeys: int = 9000) -> Dict[str, Any]:
    return _generate_dataset_impl(seed, n_journeys, None)


# ============================================================================
# Schema validation and canonical cleaning
# ============================================================================

def validate_events(raw: pd.DataFrame, now: Optional[datetime] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    now = now or utcnow()
    clean: List[Dict[str, Any]] = []
    quarantine: List[Dict[str, Any]] = []
    seen_ids = set()

    if raw.empty:
        empty = raw.copy()
        empty["quarantine_reasons"] = []
        return raw.copy(), empty

    for _, row in raw.iterrows():
        r = row.to_dict()
        issues: List[str] = []
        warnings: List[str] = []

        ts = parse_utc(r.get("timestamp_utc"))
        if ts is None:
            issues.append("bad_timestamp")
        else:
            r["timestamp_utc"] = ts
            if ts > pd.Timestamp(now) + pd.Timedelta(minutes=5):
                issues.append("future_timestamp")
            if ts < pd.Timestamp(now) - pd.Timedelta(days=180):
                issues.append("too_old")

        platform = r.get("platform")
        if is_missing(platform) or str(platform) not in PLATFORMS:
            issues.append("missing_or_unknown_platform")
        else:
            r["platform"] = str(platform)
            r["channel"] = str(platform)

        event_id = r.get("event_id")
        if event_id in seen_ids:
            issues.append("duplicate_event_id")
        else:
            seen_ids.add(event_id)

        cost_raw = r.get("cost_micros")
        if is_missing(cost_raw):
            issues.append("negative_or_missing_cost")
            cost_val = np.nan
        else:
            try:
                cost_val = float(cost_raw)
            except Exception:
                cost_val = np.nan
            if math.isnan(cost_val) or cost_val < 0:
                issues.append("negative_or_missing_cost")
            else:
                r["cost_micros"] = int(cost_val)
                if cost_val > to_micros(1000):
                    warnings.append("high_cost_outlier")

        ing = parse_utc(r.get("ingestion_time"))
        if ing is None:
            warnings.append("missing_ingestion_time")
            ing = ts if ts is not None else pd.Timestamp(now)
        r["ingestion_time"] = ing

        if ts is not None and (ing - ts) > pd.Timedelta(hours=48):
            warnings.append("late_arriving")

        if str(r.get("schema_version")) != SCHEMA_VERSION:
            issues.append("schema_version_mismatch")

        consent, consent_missing = to_bool(r.get("consent_flag"), False)
        if consent_missing:
            warnings.append("missing_consent_default_false")
        r["consent_flag"] = consent

        fraud, fraud_missing = to_bool(r.get("fraud_flag"), False)
        if fraud_missing:
            warnings.append("missing_fraud_default_false")
        r["fraud_flag"] = fraud
        if fraud:
            warnings.append("fraud_suspect")

        r["data_quality_flag"] = ",".join(warnings)

        if issues:
            r["quarantine_reasons"] = ",".join(issues)
            quarantine.append(r)
        else:
            clean.append(r)

    clean_df = pd.DataFrame(clean, columns=EVENT_FIELDS)
    quar_df = pd.DataFrame(quarantine, columns=EVENT_FIELDS + ["quarantine_reasons"])
    return clean_df, quar_df


def validate_conversions(raw: pd.DataFrame, now: Optional[datetime] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    now = now or utcnow()
    clean: List[Dict[str, Any]] = []
    quarantine: List[Dict[str, Any]] = []
    seen_ids = set()

    if raw.empty:
        empty = raw.copy()
        empty["quarantine_reasons"] = []
        return raw.copy(), empty

    for _, row in raw.iterrows():
        r = row.to_dict()
        issues: List[str] = []
        warnings: List[str] = []

        cid = r.get("conversion_id")
        if cid in seen_ids:
            issues.append("duplicate_conversion_id")
        else:
            seen_ids.add(cid)

        ct = parse_utc(r.get("conversion_time_utc"))
        if ct is None:
            issues.append("missing_or_bad_conversion_time")
        else:
            r["conversion_time_utc"] = ct
            if ct > pd.Timestamp(now) + pd.Timedelta(minutes=5):
                issues.append("future_conversion_time")
            if ct < pd.Timestamp(now) - pd.Timedelta(days=180):
                issues.append("too_old_conversion")

        rev_raw = r.get("revenue_micros")
        if is_missing(rev_raw):
            issues.append("missing_revenue")
            rev_val = np.nan
        else:
            try:
                rev_val = float(rev_raw)
            except Exception:
                rev_val = np.nan
            if math.isnan(rev_val) or rev_val < 0:
                issues.append("negative_or_invalid_revenue")
            else:
                r["revenue_micros"] = int(rev_val)
                if rev_val > to_micros(50000):
                    warnings.append("high_revenue_outlier")

        margin_raw = r.get("margin_micros")
        if is_missing(margin_raw):
            margin_val = float(rev_val) * 0.4 if not math.isnan(rev_val) else 0.0
            warnings.append("missing_margin_imputed")
        else:
            try:
                margin_val = float(margin_raw)
            except Exception:
                margin_val = 0.0
                warnings.append("bad_margin_default_zero")
        r["margin_micros"] = int(max(0.0, margin_val))

        ing = parse_utc(r.get("ingestion_time"))
        if ing is None:
            warnings.append("missing_ingestion_time")
            ing = ct if ct is not None else pd.Timestamp(now)
        r["ingestion_time"] = ing
        if ct is not None and (ing - ct) > pd.Timedelta(hours=48):
            warnings.append("late_arriving")

        if str(r.get("schema_version")) != SCHEMA_VERSION:
            issues.append("schema_version_mismatch")

        consent, consent_missing = to_bool(r.get("consent_flag"), False)
        if consent_missing:
            warnings.append("missing_consent_default_false")
        r["consent_flag"] = consent

        fraud, fraud_missing = to_bool(r.get("fraud_flag"), False)
        if fraud_missing:
            warnings.append("missing_fraud_default_false")
        r["fraud_flag"] = fraud
        if fraud:
            warnings.append("fraud_suspect")

        newc, _ = to_bool(r.get("new_customer_flag"), False)
        r["new_customer_flag"] = newc

        r["data_quality_flag"] = ",".join(warnings)

        if issues:
            r["quarantine_reasons"] = ",".join(issues)
            quarantine.append(r)
        else:
            clean.append(r)

    clean_df = pd.DataFrame(clean, columns=CONVERSION_FIELDS)
    quar_df = pd.DataFrame(quarantine, columns=CONVERSION_FIELDS + ["quarantine_reasons"])
    return clean_df, quar_df


def validate_journeys(raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    clean: List[Dict[str, Any]] = []
    quarantine: List[Dict[str, Any]] = []
    seen_ids = set()

    if raw.empty:
        empty = raw.copy()
        empty["channels"] = []
        empty["quarantine_reasons"] = []
        return empty, empty

    for _, row in raw.iterrows():
        r = row.to_dict()
        issues: List[str] = []
        warnings: List[str] = []

        jid = r.get("journey_id")
        if jid in seen_ids:
            issues.append("duplicate_journey_id")
        else:
            seen_ids.add(jid)

        try:
            channels = json.loads(r.get("channels_json") or "[]")
            if not isinstance(channels, list):
                raise ValueError
            channels = collapse_sequence([str(c) for c in channels])
            r["channels"] = channels
        except Exception:
            issues.append("bad_channels_json")
            r["channels"] = []

        terminal = str(r.get("terminal") or "null").lower()
        if terminal not in ("conversion", "null"):
            issues.append("bad_terminal")
            terminal = "null"
        r["terminal"] = terminal

        ct = parse_utc(r.get("conversion_time"))
        if ct is None:
            if terminal == "conversion":
                issues.append("missing_conversion_time")
            else:
                warnings.append("missing_null_journey_time")
            r["conversion_time"] = pd.NaT
        else:
            r["conversion_time"] = ct

        if str(r.get("schema_version")) != SCHEMA_VERSION:
            issues.append("schema_version_mismatch")

        consent, consent_missing = to_bool(r.get("consent_flag"), False)
        if consent_missing:
            warnings.append("missing_consent_default_false")
        r["consent_flag"] = consent

        r["data_quality_flag"] = ",".join(warnings)

        if issues:
            r["quarantine_reasons"] = ",".join(issues)
            quarantine.append(r)
        else:
            clean.append(r)

    clean_df = pd.DataFrame(clean)
    if not clean_df.empty:
        clean_df = clean_df.reindex(columns=JOURNEY_FIELDS + ["channels"])
    else:
        clean_df = pd.DataFrame(columns=JOURNEY_FIELDS + ["channels"])

    quar_df = pd.DataFrame(quarantine)
    if not quar_df.empty:
        quar_df = quar_df.reindex(columns=JOURNEY_FIELDS + ["channels", "quarantine_reasons"])
    else:
        quar_df = pd.DataFrame(columns=JOURNEY_FIELDS + ["channels", "quarantine_reasons"])

    return clean_df, quar_df


@st.cache_data(show_spinner="Validating synthetic data contracts...")
def load_and_validate(seed: int = SEED):
    ds = generate_dataset(seed=seed, n_journeys=9000)
    ev_c, ev_q = validate_events(ds["events"])
    co_c, co_q = validate_conversions(ds["conversions"])
    jo_c, jo_q = validate_journeys(ds["journeys"])
    return ds, ev_c, ev_q, co_c, co_q, jo_c, jo_q


# ============================================================================
# SQLite persistence and audit history
# ============================================================================

@st.cache_resource(show_spinner=False)
def get_db_conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def _df_for_sql(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        try:
            if pd.api.types.is_datetime64_any_dtype(out[col]):
                if getattr(out[col].dt, "tz", None) is not None:
                    out[col] = out[col].dt.tz_convert("UTC")
                out[col] = out[col].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            out[col] = out[col].astype(str)
    return out


def persist_dataset(
    conn: sqlite3.Connection,
    ev_c: pd.DataFrame,
    co_c: pd.DataFrame,
    jo_c: pd.DataFrame,
    ev_q: pd.DataFrame,
    co_q: pd.DataFrame,
    jo_q: pd.DataFrame,
) -> None:
    _df_for_sql(ev_c).to_sql("events_clean", conn, if_exists="replace", index=False)
    _df_for_sql(co_c).to_sql("conversions_clean", conn, if_exists="replace", index=False)
    _df_for_sql(jo_c.drop(columns=["channels"], errors="ignore")).to_sql(
        "journeys_clean", conn, if_exists="replace", index=False
    )
    _df_for_sql(ev_q).to_sql("quarantine_events", conn, if_exists="replace", index=False)
    _df_for_sql(co_q).to_sql("quarantine_conversions", conn, if_exists="replace", index=False)
    _df_for_sql(jo_q.drop(columns=["channels"], errors="ignore")).to_sql(
        "quarantine_journeys", conn, if_exists="replace", index=False
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            actor TEXT,
            action TEXT,
            details_json TEXT,
            approval_status TEXT
        )
        """
    )
    conn.commit()


def log_audit(action: str, details: Optional[Dict[str, Any]] = None, approval_status: str = "not_required") -> None:
    try:
        conn = get_db_conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                actor TEXT,
                action TEXT,
                details_json TEXT,
                approval_status TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO audit_log (ts, actor, action, details_json, approval_status) VALUES (?, ?, ?, ?, ?)",
            (
                utcnow().isoformat(),
                "local_demo_user",
                action,
                json_dumps(details or {}),
                approval_status,
            ),
        )
        conn.commit()
    except Exception:
        pass


# ============================================================================
# Attribution path construction
# ============================================================================

def build_conversion_paths(
    events: pd.DataFrame,
    conversions: pd.DataFrame,
    window_days: int = 7,
    include_view: bool = True,
    require_consent: bool = True,
) -> pd.DataFrame:
    cols = [
        "conversion_id",
        "user_pseudo_id",
        "conversion_time_utc",
        "revenue_micros",
        "margin_micros",
        "channels",
        "lags_hours",
        "event_types",
        "has_click",
        "path_length",
    ]

    if conversions.empty:
        return pd.DataFrame(columns=cols)

    conv = conversions.copy()
    ev = events.copy()

    if require_consent:
        conv = conv[conv["consent_flag"] == True]  # noqa: E712
        ev = ev[ev["consent_flag"] == True]  # noqa: E712

    if not include_view:
        ev = ev[ev["event_type"] == "click"]

    ev_by_user: Dict[str, pd.DataFrame] = {}
    if not ev.empty:
        for uid, grp in ev.groupby("user_pseudo_id"):
            ev_by_user[str(uid)] = grp.sort_values("timestamp_utc")

    rows: List[Dict[str, Any]] = []
    for _, c in conv.iterrows():
        ct = c["conversion_time_utc"]
        uid = str(c["user_pseudo_id"])
        grp = ev_by_user.get(uid)

        channels: List[str] = []
        lags: List[float] = []
        etypes: List[str] = []

        if grp is not None and not grp.empty:
            mask = (grp["timestamp_utc"] <= ct) & (
                grp["timestamp_utc"] >= ct - pd.Timedelta(days=int(window_days))
            )
            sel = grp[mask]
            last_platform = None
            for _, e in sel.iterrows():
                p = str(e["platform"])
                if p != last_platform:
                    channels.append(p)
                    lags.append(float((ct - e["timestamp_utc"]).total_seconds() / 3600.0))
                    etypes.append(str(e["event_type"]))
                    last_platform = p

        if not channels:
            channels = ["Direct"]
            lags = [0.0]
            etypes = ["direct"]

        rows.append(
            {
                "conversion_id": c["conversion_id"],
                "user_pseudo_id": uid,
                "conversion_time_utc": ct,
                "revenue_micros": int(c["revenue_micros"]),
                "margin_micros": int(c["margin_micros"]),
                "channels": channels,
                "lags_hours": lags,
                "event_types": etypes,
                "has_click": any(x == "click" for x in etypes),
                "path_length": len(channels),
            }
        )

    return pd.DataFrame(rows, columns=cols)


# ============================================================================
# Rule-based attribution models
# ============================================================================

def attribute_with_model(
    paths: pd.DataFrame,
    model: str,
    half_life_days: float = 2.0,
) -> pd.DataFrame:
    cols = [
        "conversion_id",
        "platform",
        "credit_share",
        "credited_revenue_micros",
        "credited_margin_micros",
        "model",
    ]
    if paths.empty:
        return pd.DataFrame(columns=cols)

    rows: List[Dict[str, Any]] = []

    for _, p in paths.iterrows():
        channels = list(p["channels"]) or ["Direct"]
        lags = list(p["lags_hours"]) or [0.0]
        etypes = list(p["event_types"]) or ["direct"]
        rev = float(p["revenue_micros"])
        margin = float(p["margin_micros"])
        n = len(channels)
        shares = [0.0] * n

        if model == "last_click":
            click_idxs = [i for i, e in enumerate(etypes) if e == "click"]
            idx = click_idxs[-1] if click_idxs else n - 1
            shares[idx] = 1.0

        elif model == "first_click":
            click_idxs = [i for i, e in enumerate(etypes) if e == "click"]
            idx = click_idxs[0] if click_idxs else 0
            shares[idx] = 1.0

        elif model == "linear":
            shares = [1.0 / n for _ in range(n)]

        elif model == "time_decay":
            hl = max(0.1, float(half_life_days))
            weights = [0.5 ** ((lag / 24.0) / hl) for lag in lags]
            total = sum(weights)
            if total <= 0:
                shares = [1.0 / n for _ in range(n)]
            else:
                shares = [w / total for w in weights]

        elif model == "position":
            if n == 1:
                shares = [1.0]
            elif n == 2:
                shares = [0.5, 0.5]
            else:
                middle_share = 0.2 / max(1, n - 2)
                shares = [0.4] + [middle_share] * (n - 2) + [0.4]

        else:
            shares = [1.0 / n for _ in range(n)]

        shares = [max(0.0, float(s)) for s in shares]
        ssum = sum(shares)
        if ssum <= 0:
            shares = [1.0 / n for _ in range(n)]
        else:
            shares = [s / ssum for s in shares]

        for ch, sh in zip(channels, shares):
            rows.append(
                {
                    "conversion_id": p["conversion_id"],
                    "platform": ch,
                    "credit_share": sh,
                    "credited_revenue_micros": rev * sh,
                    "credited_margin_micros": margin * sh,
                    "model": model,
                }
            )

    return pd.DataFrame(rows, columns=cols)


def markov_attribute_paths(paths: pd.DataFrame, importance: Dict[str, float]) -> pd.DataFrame:
    cols = [
        "conversion_id",
        "platform",
        "credit_share",
        "credited_revenue_micros",
        "credited_margin_micros",
        "model",
    ]
    if paths.empty:
        return pd.DataFrame(columns=cols)

    rows: List[Dict[str, Any]] = []
    for _, p in paths.iterrows():
        channels = list(p["channels"]) or ["Direct"]
        weights = [max(0.0, float(importance.get(c, 0.001))) for c in channels]
        total = sum(weights)
        if total <= 0:
            shares = [1.0 / len(channels) for _ in channels]
        else:
            shares = [w / total for w in weights]

        rev = float(p["revenue_micros"])
        margin = float(p["margin_micros"])
        for ch, sh in zip(channels, shares):
            rows.append(
                {
                    "conversion_id": p["conversion_id"],
                    "platform": ch,
                    "credit_share": sh,
                    "credited_revenue_micros": rev * sh,
                    "credited_margin_micros": margin * sh,
                    "model": "markov",
                }
            )
    return pd.DataFrame(rows, columns=cols)


# ============================================================================
# Markov removal-effect model
# ============================================================================

def compute_markov_importance(
    journeys_df: pd.DataFrame,
    channels: Optional[List[str]] = None,
    n_bootstrap: int = 20,
    seed: int = SEED,
    min_paths: int = 300,
    min_conv_prob: float = 0.01,
    require_consent: bool = True,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    channels = channels or PLATFORMS
    df = journeys_df.copy()

    if require_consent and "consent_flag" in df.columns:
        df = df[df["consent_flag"] == True]  # noqa: E712

    data: List[Tuple[List[str], str]] = []
    for _, r in df.iterrows():
        seq = collapse_sequence(list(r.get("channels") or []))
        term = "CONVERSION" if str(r.get("terminal")).lower() == "conversion" else "NULL"
        data.append((seq, term))

    def _importance_from_sample(sample: List[Tuple[List[str], str]]) -> Dict[str, Any]:
        def absorption(removed: Optional[str] = None) -> float:
            counts: Dict[str, Dict[str, float]] = {
                s: {} for s in ["START"] + list(channels) + ["CONVERSION", "NULL"]
            }

            for seq, term in sample:
                s = list(seq)
                t = term
                if removed is not None and removed in s:
                    idx = s.index(removed)
                    s = s[:idx]
                    t = "NULL"
                full = ["START"] + s + [t]
                for a, b in zip(full, full[1:]):
                    counts.setdefault(a, {})
                    counts[a][b] = counts[a].get(b, 0.0) + 1.0

            transient = ["START"] + [c for c in channels if c != removed]
            idx_map = {s: i for i, s in enumerate(transient)}
            m = len(transient)
            A = np.zeros((m, m), dtype=float)
            b = np.zeros(m, dtype=float)

            for s in transient:
                i = idx_map[s]
                A[i, i] = 1.0
                total = sum(counts.get(s, {}).values())
                if total <= 0:
                    continue
                for nxt, cnt in counts.get(s, {}).items():
                    prob = cnt / total
                    if nxt == "CONVERSION":
                        b[i] += prob
                    elif nxt == "NULL":
                        pass
                    elif nxt in idx_map:
                        A[i, idx_map[nxt]] -= prob
                    else:
                        pass

            try:
                h = np.linalg.solve(A, b)
            except Exception:
                h = np.linalg.lstsq(A, b, rcond=None)[0]

            start_idx = idx_map.get("START", 0)
            return float(max(0.0, min(1.0, h[start_idx])))

        full_p = absorption(None)
        effects: Dict[str, float] = {}
        for c in channels:
            rem_p = absorption(c)
            if full_p <= 1e-9:
                eff = 0.0
            else:
                eff = max(0.0, 1.0 - rem_p / full_p)
            effects[c] = eff

        eff_sum = sum(effects.values())
        if eff_sum <= 1e-9:
            vec = np.ones(len(channels), dtype=float) / len(channels)
        else:
            vec = np.array([effects[c] / eff_sum for c in channels], dtype=float)

        return {
            "vector": vec,
            "full_p": full_p,
            "effects": effects,
        }

    if not data:
        importance = {c: 1.0 / len(channels) for c in channels}
        meta = {
            "n_journeys": 0,
            "n_conversions": 0,
            "conversion_probability": 0.0,
            "consent_coverage": 0.0,
            "mean_bootstrap_std": 1.0,
            "confidence": "low",
            "fallback_recommended": True,
            "ci": {c: (0.0, 0.0) for c in channels},
            "full_conversion_probability": 0.0,
        }
        return importance, meta

    base = _importance_from_sample(data)
    vec = base["vector"]
    importance = {c: float(vec[i]) for i, c in enumerate(channels)}

    n_conv = sum(1 for _, t in data if t == "CONVERSION")
    full_p = float(base["full_p"])
    consent_cov = float(df["consent_flag"].mean()) if "consent_flag" in df.columns and not df.empty else 0.0

    boot_vectors: List[np.ndarray] = []
    if len(data) >= 2 and n_bootstrap > 0:
        rng = np.random.default_rng(seed)
        nb = max(5, min(int(n_bootstrap), 50))
        for _ in range(nb):
            idx = rng.integers(0, len(data), len(data))
            sample = [data[i] for i in idx]
            try:
                boot_vectors.append(_importance_from_sample(sample)["vector"])
            except Exception:
                pass

    if boot_vectors:
        arr = np.vstack(boot_vectors)
        std = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(vec)
        lower = np.percentile(arr, 2.5, axis=0)
        upper = np.percentile(arr, 97.5, axis=0)
    else:
        std = np.zeros_like(vec)
        lower = vec.copy()
        upper = vec.copy()

    ci = {c: (float(lower[i]), float(upper[i])) for i, c in enumerate(channels)}
    mean_std = float(np.mean(std))

    if (
        len(data) < min_paths
        or full_p < min_conv_prob
        or consent_cov < 0.70
        or mean_std > 0.08
    ):
        confidence = "low"
    elif mean_std > 0.04 or consent_cov < 0.85:
        confidence = "medium"
    else:
        confidence = "high"

    meta = {
        "n_journeys": len(data),
        "n_conversions": n_conv,
        "conversion_probability": full_p,
        "consent_coverage": consent_cov,
        "mean_bootstrap_std": mean_std,
        "confidence": confidence,
        "fallback_recommended": confidence == "low",
        "ci": ci,
        "full_conversion_probability": full_p,
    }
    return importance, meta


def prepare_journeys_for_model(
    journeys_df: pd.DataFrame,
    conversions_df: pd.DataFrame,
    require_consent: bool = True,
) -> pd.DataFrame:
    df = journeys_df.copy()
    valid_ids = set(conversions_df["conversion_id"].dropna().astype(str))
    df = df[(df["terminal"] != "conversion") | (df["conversion_id"].astype(str).isin(valid_ids))]
    if require_consent and "consent_flag" in df.columns:
        df = df[df["consent_flag"] == True]  # noqa: E712
    return df


# ============================================================================
# Aggregation, decisions, and data quality
# ============================================================================

def channel_credit_shares(credits_df: pd.DataFrame) -> pd.Series:
    if credits_df.empty:
        return pd.Series(0.0, index=PLATFORMS)
    rev = credits_df.groupby("platform")["credited_revenue_micros"].sum()
    rev = rev.reindex(PLATFORMS).fillna(0.0)
    total = float(rev.sum())
    if total <= 0:
        return pd.Series(0.0, index=PLATFORMS)
    return rev / total


def aggregate_performance(
    events: pd.DataFrame,
    conversions: pd.DataFrame,
    credits_df: pd.DataFrame,
    model_key: str,
    settings: Dict[str, Any],
    markov_meta: Dict[str, Any],
    last_shares: Optional[pd.Series] = None,
) -> pd.DataFrame:
    cost_events = events[events["consent_flag"] == True] if settings["require_consent"] else events  # noqa: E712
    costs = cost_events.groupby("platform")["cost_micros"].sum().reindex(PLATFORMS).fillna(0)
    attr_rev = credits_df.groupby("platform")["credited_revenue_micros"].sum().reindex(PLATFORMS).fillna(0)
    attr_margin = credits_df.groupby("platform")["credited_margin_micros"].sum().reindex(PLATFORMS).fillna(0)
    attr_conv = credits_df.groupby("platform")["credit_share"].sum().reindex(PLATFORMS).fillna(0)

    df = pd.DataFrame(
        {
            "platform": PLATFORMS,
            "cost_usd": [from_micros(costs.get(p, 0)) for p in PLATFORMS],
            "attributed_revenue_usd": [from_micros(attr_rev.get(p, 0)) for p in PLATFORMS],
            "attributed_margin_usd": [from_micros(attr_margin.get(p, 0)) for p in PLATFORMS],
            "attributed_conversions": [float(attr_conv.get(p, 0)) for p in PLATFORMS],
        }
    )

    df["roas"] = df.apply(lambda r: safe_div(r["attributed_revenue_usd"], r["cost_usd"]), axis=1)
    df["margin_roas"] = df.apply(lambda r: safe_div(r["attributed_margin_usd"], r["cost_usd"]), axis=1)
    df["cac_usd"] = df.apply(lambda r: safe_div(r["cost_usd"], r["attributed_conversions"]), axis=1)

    shares = channel_credit_shares(credits_df)
    df["attributed_revenue_share"] = df["platform"].map(shares).astype(float)

    if last_shares is None:
        last_shares = shares
    df["disagreement_vs_last_click"] = df["platform"].map(
        lambda p: abs(float(shares.get(p, 0.0)) - float(last_shares.get(p, 0.0)))
    )

    ci = markov_meta.get("ci", {})

    def _confidence(p: str) -> float:
        if p in ci:
            lo, hi = ci[p]
            width = max(0.0, float(hi) - float(lo))
            return clamp(1.0 - min(1.0, width * 4.0), 0.05, 0.99)
        return 0.50

    df["confidence_score"] = df["platform"].map(_confidence).astype(float)
    if markov_meta.get("fallback_used") or markov_meta.get("confidence") == "low":
        df["confidence_score"] = np.minimum(df["confidence_score"], 0.35)

    df["model"] = model_key
    return df


def make_recommendations(
    perf: pd.DataFrame,
    markov_meta: Dict[str, Any],
    dq: Dict[str, Any],
    settings: Dict[str, Any],
) -> pd.DataFrame:
    target = float(settings["target_roas"])
    min_conf_map = {"high": 0.70, "medium": 0.45, "low": 0.20}
    min_conf = min_conf_map.get(settings["min_confidence"], 0.45)
    safe_mode = bool(settings.get("safe_mode", False))

    rows: List[Dict[str, Any]] = []
    for _, r in perf.iterrows():
        platform = r["platform"]
        cost = float(r["cost_usd"])
        roas = float(r["roas"])
        conf = float(r["confidence_score"])
        disagree = float(r["disagreement_vs_last_click"])

        risk_flags: List[str] = []
        if disagree > 0.25:
            risk_flags.append("high_baseline_disagreement")
        if dq.get("consent_coverage", 1.0) < 0.85:
            risk_flags.append("low_consent_coverage")
        if dq.get("quarantine_rate", 0.0) > 0.05:
            risk_flags.append("elevated_quarantine_rate")
        if dq.get("fraud_rate", 0.0) > 0.02:
            risk_flags.append("fraud_signal")
        if markov_meta.get("fallback_used"):
            risk_flags.append("model_fallback_active")

        if safe_mode:
            rec = "Hold"
            change = 0.0
            approval = False
            rationale = "Safe mode enabled; all spend changes suppressed."
        elif platform == "Direct" or cost < 1.0:
            rec = "Hold"
            change = 0.0
            approval = False
            rationale = "Non-actionable or insufficient cost signal."
        elif conf < min_conf:
            rec = "Investigate"
            change = 0.0
            approval = False
            rationale = f"Confidence {conf:.2f} below action threshold {min_conf:.2f}."
        elif roas >= target * 1.15:
            rec = "Increase"
            change = 8.0
            approval = True
            rationale = f"ROAS {roas:.2f} exceeds target {target:.2f} with acceptable confidence."
        elif roas <= target * 0.85:
            rec = "Decrease"
            change = -8.0
            approval = True
            rationale = f"ROAS {roas:.2f} below target {target:.2f} with acceptable confidence."
        else:
            rec = "Hold"
            change = 0.0
            approval = False
            rationale = "Performance near target; no action recommended."

        rows.append(
            {
                "platform": platform,
                "recommendation": rec,
                "suggested_change_pct": change,
                "estimated_daily_budget_change_usd": cost * change / 100.0,
                "requires_human_approval": approval,
                "rationale": rationale,
                "risk_flags": ", ".join(risk_flags) if risk_flags else "",
                "confidence_score": conf,
                "roas": roas,
                "cost_usd": cost,
            }
        )

    return pd.DataFrame(rows)


def compute_data_quality(
    raw_events: pd.DataFrame,
    clean_events: pd.DataFrame,
    raw_conversions: pd.DataFrame,
    clean_conversions: pd.DataFrame,
    quar_events: pd.DataFrame,
    quar_conversions: pd.DataFrame,
    paths: pd.DataFrame,
    settings: Dict[str, Any],
) -> Dict[str, Any]:
    total_ev = max(1, len(raw_events))
    total_co = max(1, len(raw_conversions))

    quarantine_rate = len(quar_events) / total_ev
    conversion_quarantine_rate = len(quar_conversions) / total_co
    consent_cov = float(clean_events["consent_flag"].mean()) if not clean_events.empty else 0.0
    late_rate = (
        float(clean_events["data_quality_flag"].str.contains("late", na=False).mean())
        if not clean_events.empty
        else 0.0
    )
    fraud_rate = float(clean_events["fraud_flag"].mean()) if not clean_events.empty else 0.0
    outlier_rate = (
        float((clean_conversions["revenue_micros"] > to_micros(50000)).mean())
        if not clean_conversions.empty
        else 0.0
    )
    path_cov = (
        float(paths["channels"].apply(lambda x: list(x) != ["Direct"]).mean())
        if not paths.empty
        else 0.0
    )

    score = (
        0.35 * consent_cov
        + 0.25 * (1.0 - quarantine_rate)
        + 0.20 * path_cov
        + 0.10 * (1.0 - late_rate)
        + 0.10 * (1.0 - fraud_rate)
    )

    return {
        "score": clamp(score),
        "consent_coverage": consent_cov,
        "quarantine_rate": quarantine_rate,
        "conversion_quarantine_rate": conversion_quarantine_rate,
        "late_rate": late_rate,
        "fraud_rate": fraud_rate,
        "outlier_rate": outlier_rate,
        "path_coverage": path_cov,
        "raw_event_count": len(raw_events),
        "clean_event_count": len(clean_events),
        "raw_conversion_count": len(raw_conversions),
        "clean_conversion_count": len(clean_conversions),
    }


# ============================================================================
# Main attribution orchestration
# ============================================================================

def ensure_attribution_store(bundle, settings: Dict[str, Any]) -> Dict[str, Any]:
    ds, ev_c, ev_q, co_c, co_q, jo_c, jo_q = bundle
    sh = json_dumps(settings)

    if st.session_state.get("attr_hash") == sh and "attr_store" in st.session_state:
        return st.session_state["attr_store"]

    paths = build_conversion_paths(
        ev_c,
        co_c,
        window_days=int(settings["window_days"]),
        include_view=bool(settings["include_view"]),
        require_consent=bool(settings["require_consent"]),
    )

    journeys_model = prepare_journeys_for_model(jo_c, co_c, settings["require_consent"])
    boot = 5 if settings["force_fallback"] else 20
    importance, markov_meta = compute_markov_importance(
        journeys_model,
        channels=PLATFORMS,
        n_bootstrap=boot,
        seed=SEED,
        min_paths=300,
        min_conv_prob=0.01,
        require_consent=settings["require_consent"],
    )

    use_markov = (
        not settings["force_fallback"]
        and markov_meta.get("confidence") != "low"
        and markov_meta.get("consent_coverage", 0.0) >= 0.70
    )

    if use_markov:
        markov_credits = markov_attribute_paths(paths, importance)
        markov_meta["fallback_used"] = False
        markov_meta["fallback_reason"] = ""
    else:
        markov_credits = attribute_with_model(paths, "linear", settings["half_life_days"])
        markov_meta["fallback_used"] = True
        reasons = []
        if settings["force_fallback"]:
            reasons.append("force_fallback_enabled")
        if markov_meta.get("confidence") == "low":
            reasons.append("low_model_confidence")
        if markov_meta.get("consent_coverage", 0.0) < 0.70:
            reasons.append("low_consent_coverage")
        markov_meta["fallback_reason"] = ", ".join(reasons) or "deterministic_fallback"

    credits = {
        "last_click": attribute_with_model(paths, "last_click"),
        "first_click": attribute_with_model(paths, "first_click"),
        "linear": attribute_with_model(paths, "linear", settings["half_life_days"]),
        "time_decay": attribute_with_model(paths, "time_decay", settings["half_life_days"]),
        "position": attribute_with_model(paths, "position", settings["half_life_days"]),
        "markov": markov_credits,
    }

    selected_key = settings["model"]
    last_shares = channel_credit_shares(credits["last_click"])
    perf = aggregate_performance(
        ev_c,
        co_c,
        credits[selected_key],
        selected_key,
        settings,
        markov_meta,
        last_shares,
    )
    dq = compute_data_quality(ds["events"], ev_c, ds["conversions"], co_c, ev_q, co_q, paths, settings)
    recs = make_recommendations(perf, markov_meta, dq, settings)

    store = {
        "paths": paths,
        "credits": credits,
        "performance": perf,
        "recommendations": recs,
        "data_quality": dq,
        "markov_importance": importance,
        "markov_meta": markov_meta,
        "settings_hash": sh,
    }

    st.session_state["attr_store"] = store
    st.session_state["attr_hash"] = sh

    log_audit(
        "model_run",
        {
            "settings": settings,
            "fallback_used": markov_meta.get("fallback_used"),
            "fallback_reason": markov_meta.get("fallback_reason"),
            "confidence": markov_meta.get("confidence"),
            "data_quality_score": dq.get("score"),
        },
    )

    return store


# ============================================================================
# Live simulation
# ============================================================================

def ensure_live_state(store: Dict[str, Any], bundle) -> None:
    if "live_initialized" in st.session_state:
        return

    ds = bundle[0]
    truth = ds["truth"]
    act = {p: float(truth.get(p, 0.0)) for p in ACTIONABLE_PLATFORMS}
    s = sum(act.values()) or 1.0
    st.session_state.live_presence = {k: v / s for k, v in act.items()}
    st.session_state.live_tick = 0
    st.session_state.live_history = []
    st.session_state.live_events = []
    st.session_state.live_alerts = []
    st.session_state.live_cumulative = {"revenue_usd": 0.0, "cost_usd": 0.0, "conversions": 0}
    st.session_state.live_attr_revenue = {p: 0.0 for p in PLATFORMS}
    st.session_state.live_initialized = True


def reset_live_state() -> None:
    for k in [
        "live_initialized",
        "live_tick",
        "live_history",
        "live_events",
        "live_alerts",
        "live_cumulative",
        "live_attr_revenue",
        "live_presence",
        "demo_video",
        "demo_video_error",
        "live_seeded",
    ]:
        st.session_state.pop(k, None)


def get_live_model_shares(store: Dict[str, Any], bundle) -> Dict[str, float]:
    perf = store["performance"]
    total = float(perf["attributed_revenue_usd"].sum())
    shares = {p: 0.0 for p in PLATFORMS}
    if total > 0:
        for _, r in perf.iterrows():
            shares[str(r["platform"])] = float(r["attributed_revenue_usd"]) / total
    else:
        shares = {p: float(bundle[0]["truth"].get(p, 0.0)) for p in PLATFORMS}
    s = sum(shares.values()) or 1.0
    return {k: v / s for k, v in shares.items()}


def tick_live(store: Dict[str, Any], bundle, speed: str = "normal") -> None:
    ensure_live_state(store, bundle)
    mult = {"slow": 0.5, "normal": 1.0, "fast": 2.0}.get(speed, 1.0)
    tick = int(st.session_state.live_tick) + 1
    st.session_state.live_tick = tick

    rng = np.random.default_rng(SEED + tick * 7919)
    shares = get_live_model_shares(store, bundle)
    presence = st.session_state.live_presence
    platforms = list(presence.keys())
    pvals = np.array([presence[p] for p in platforms], dtype=float)
    pvals /= pvals.sum()

    n_events = int(rng.poisson(6.0 * mult))
    cost_delta = 0.0
    new_events: List[Dict[str, Any]] = []

    for _ in range(max(0, n_events)):
        p = str(rng.choice(platforms, p=pvals))
        et = str(rng.choice(["click", "view", "video_view", "impression"], p=[0.45, 0.25, 0.15, 0.15]))
        base = CPC_USD.get(p, 0.5) if et == "click" else CPV_USD.get(p, 0.04)
        cost = max(0.01, float(rng.lognormal(math.log(max(base, 0.01)), 0.35)))
        cost_delta += cost
        new_events.append(
            {
                "tick": tick,
                "timestamp": utcnow().strftime("%H:%M:%S"),
                "platform": p,
                "event_type": et,
                "cost_usd": cost,
            }
        )

    st.session_state.live_events = (new_events + st.session_state.live_events)[:60]

    conv_delta = 0
    rev_delta = 0.0
    if rng.random() < 0.38 * mult:
        rev_delta = float(rng.lognormal(math.log(95.0), 0.45))
        if rng.random() < 0.01:
            rev_delta *= 8.0
        conv_delta = 1
        for p, sh in shares.items():
            noise = max(0.0, float(rng.normal(1.0, 0.08)))
            st.session_state.live_attr_revenue[p] = float(
                st.session_state.live_attr_revenue.get(p, 0.0) + rev_delta * max(0.0, sh) * noise
            )

    alert = None
    if tick % 41 == 0:
        alert = {
            "tick": tick,
            "type": "cost_spike",
            "message": "Simulated Meta Feed CPM spike +18%; recommendation capped pending review.",
        }
        cost_delta *= 1.18
    if tick % 53 == 0:
        alert = {
            "tick": tick,
            "type": "consent_drop",
            "message": "Simulated consent coverage dropped; model confidence reduced.",
        }

    if alert:
        st.session_state.live_alerts = [alert] + st.session_state.live_alerts[:25]

    cum = st.session_state.live_cumulative
    cum["cost_usd"] = float(cum["cost_usd"]) + cost_delta
    cum["revenue_usd"] = float(cum["revenue_usd"]) + rev_delta
    cum["conversions"] = int(cum["conversions"]) + conv_delta

    quality_score = float(store["data_quality"]["score"])
    if alert and alert["type"] == "consent_drop":
        quality_score *= 0.94

    history = st.session_state.live_history
    history.append(
        {
            "tick": tick,
            "timestamp": utcnow().isoformat(),
            "revenue_delta_usd": rev_delta,
            "cost_delta_usd": cost_delta,
            "conversions_delta": conv_delta,
            "cumulative_revenue_usd": cum["revenue_usd"],
            "cumulative_cost_usd": cum["cost_usd"],
            "cumulative_conversions": cum["conversions"],
            "roas": safe_div(cum["revenue_usd"], cum["cost_usd"]),
            "cac_usd": safe_div(cum["cost_usd"], cum["conversions"]),
            "quality_score": quality_score,
            "alert": alert["type"] if alert else "",
        }
    )
    if len(history) > 180:
        history.pop(0)


_FONT_CACHE: Dict[int, Any] = {}


def get_font(size: int) -> Any:
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            f = ImageFont.truetype(name, size)
            _FONT_CACHE[size] = f
            return f
        except Exception:
            pass
    try:
        f = ImageFont.load_default(size)
    except Exception:
        f = ImageFont.load_default()
    _FONT_CACHE[size] = f
    return f


def _write_video_frames(frames_rgb: List[np.ndarray], fps: int) -> Tuple[bytes, str, str]:
    if not frames_rgb:
        raise ValueError("No frames to encode")

    h, w = frames_rgb[0].shape[:2]

    if CV2_AVAILABLE:
        candidates = [
            (".mp4", "mp4v", "video/mp4", "cross_platform_attribution_live_demo.mp4"),
            (".avi", "MJPG", "video/x-msvideo", "cross_platform_attribution_live_demo.avi"),
        ]
        for suffix, code, mime, fname in candidates:
            path = None
            try:
                tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
                path = tmp.name
                tmp.close()
                writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*code), float(fps), (w, h))
                if writer is not None and writer.isOpened():
                    for fr in frames_rgb:
                        writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
                    writer.release()
                    with open(path, "rb") as f:
                        data = f.read()
                    if len(data) > 0:
                        return data, mime, fname
            except Exception:
                pass
            finally:
                try:
                    if path and os.path.exists(path):
                        os.unlink(path)
                except Exception:
                    pass

    buf = io.BytesIO()
    imgs = [Image.fromarray(fr) for fr in frames_rgb]
    imgs[0].save(
        buf,
        format="GIF",
        append_images=imgs[1:],
        save_all=True,
        duration=int(1000.0 / max(1, fps)),
        loop=0,
    )
    return buf.getvalue(), "image/gif", "cross_platform_attribution_live_demo.gif"


def generate_demo_video(
    history: List[Dict[str, Any]],
    channel_revenue: Dict[str, float],
    model_label: str,
    confidence_label: str,
    dq_score: float,
    duration_s: float = 4.0,
    fps: int = 9,
) -> Tuple[bytes, str, str]:
    width, height = 1280, 720
    n_frames = max(12, int(duration_s * fps))

    if not history:
        history = [
            {
                "cumulative_revenue_usd": 1200.0 * (i + 1),
                "cumulative_cost_usd": 650.0 * (i + 1),
                "cumulative_conversions": int(9.5 * (i + 1)),
                "quality_score": 0.91,
            }
            for i in range(60)
        ]

    rev_hist = [float(h.get("cumulative_revenue_usd", 0.0)) for h in history]
    max_rev = max(rev_hist) if rev_hist else 1.0
    max_rev = max(1.0, max_rev)

    items = sorted(
        [(k, float(v)) for k, v in channel_revenue.items() if float(v) > 0],
        key=lambda x: x[1],
        reverse=True,
    )[:6]
    if not items:
        items = [("Google Search", 1000.0), ("Meta Feed", 750.0), ("Email", 450.0)]
    max_bar = max([v for _, v in items] + [1.0])

    frames: List[np.ndarray] = []

    for fi in range(n_frames):
        progress = (fi + 1) / float(n_frames)
        img = Image.new("RGB", (width, height), (13, 17, 23))
        d = ImageDraw.Draw(img)

        d.rectangle([0, 0, width, 92], fill=(22, 33, 48))
        d.text(
            (30, 18),
            "Cross-Platform Attribution Engine — Live Synthetic Demo",
            font=get_font(30),
            fill=(240, 244, 248),
        )
        d.text(
            (30, 58),
            f"Model: {model_label} | Confidence: {confidence_label} | Data quality: {dq_score:.0%}",
            font=get_font(18),
            fill=(160, 190, 220),
        )

        idx = min(int(progress * (len(history) - 1)), len(history) - 1)
        last = history[idx]

        kpis = [
            ("Revenue (simulated observed)", fmt_usd(last.get("cumulative_revenue_usd", 0))),
            ("Cost (simulated observed)", fmt_usd(last.get("cumulative_cost_usd", 0))),
            ("Conversions", f"{int(last.get('cumulative_conversions', 0)):,}"),
            ("ROAS", f"{float(last.get('roas', 0)):.2f}"),
        ]
        x = 30
        for title, val in kpis:
            d.rectangle([x, 112, x + 280, 208], outline=(70, 100, 130), width=2)
            d.text((x + 15, 126), title, font=get_font(16), fill=(150, 170, 190))
            d.text((x + 15, 158), val, font=get_font(30), fill=(255, 255, 255))
            x += 300

        d.rectangle([30, 238, 760, 470], outline=(50, 70, 90), width=2)
        d.text((40, 244), "Cumulative revenue trend (simulated observed)", font=get_font(16), fill=(180, 200, 220))
        limit = max(2, int(progress * len(rev_hist)))
        pts = []
        for i in range(limit):
            px = 50 + int(700.0 * (i / max(1, len(rev_hist) - 1)))
            py = 450 - int(160.0 * (rev_hist[i] / max_rev))
            pts.append((px, py))
        if len(pts) > 1:
            d.line(pts, fill=(0, 220, 160), width=3)

        d.rectangle([790, 238, 1250, 470], outline=(50, 70, 90), width=2)
        d.text((800, 244), "Attributed revenue by platform (estimated)", font=get_font(16), fill=(180, 200, 220))
        y = 280
        for name, val in items:
            bar_len = int(390.0 * (val / max_bar) * progress)
            bar_len = max(2, bar_len)
            d.text((800, y), name[:24], font=get_font(14), fill=(220, 220, 220))
            d.rectangle([800, y + 18, 800 + bar_len, y + 34], fill=(80, 160, 255))
            d.text((800 + bar_len + 5, y + 18), fmt_usd(val * progress), font=get_font(12), fill=(200, 200, 200))
            y += 45
            if y > 445:
                break

        d.text(
            (30, 640),
            "Labels: observed = simulated stream; estimated = attribution model; synthetic data only.",
            font=get_font(16),
            fill=(170, 180, 190),
        )
        d.text(
            (30, 668),
            "Governance: human approval required for spend changes. No real customer data.",
            font=get_font(16),
            fill=(170, 180, 190),
        )

        frames.append(np.array(img))

    return _write_video_frames(frames, fps)


def render_live_outputs(store: Dict[str, Any], bundle, settings: Dict[str, Any]) -> None:
    history = st.session_state.get("live_history", [])
    if not history:
        st.info("Waiting for first simulated tick...")
        return

    last = history[-1]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Revenue", fmt_usd(last.get("cumulative_revenue_usd", 0)), help="Simulated observed revenue")
    c2.metric("Cost", fmt_usd(last.get("cumulative_cost_usd", 0)), help="Simulated observed cost")
    c3.metric("Conversions", f"{int(last.get('cumulative_conversions', 0)):,}")
    c4.metric("ROAS", f"{float(last.get('roas', 0)):.2f}")
    c5.metric("Data quality", f"{float(last.get('quality_score', 0)):.0%}")

    df = pd.DataFrame(history)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["tick"],
            y=df["cumulative_revenue_usd"],
            name="Cumulative revenue",
            line=dict(color="#00dca0", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["tick"],
            y=df["cumulative_cost_usd"],
            name="Cumulative cost",
            line=dict(color="#ff7f50", width=3),
        )
    )
    fig.update_layout(
        title="Live simulated ad spend and revenue",
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", y=1.12),
        xaxis_title="Tick",
        yaxis_title="USD",
    )
    st.plotly_chart(fig, use_container_width=True)

    attr = st.session_state.get("live_attr_revenue", {})
    bar_df = pd.DataFrame(
        {
            "platform": list(attr.keys()),
            "attributed_revenue_usd": list(attr.values()),
        }
    )
    bar_df = bar_df[bar_df["attributed_revenue_usd"] > 0].sort_values("attributed_revenue_usd", ascending=False)
    if not bar_df.empty:
        fig2 = px.bar(
            bar_df,
            x="platform",
            y="attributed_revenue_usd",
            title="Live attributed revenue by platform (estimated)",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig2.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.caption("No live attributed revenue yet.")

    left, right = st.columns(2)
    with left:
        st.subheader("Recent simulated events")
        ev = pd.DataFrame(st.session_state.get("live_events", []))
        if ev.empty:
            st.caption("No events yet.")
        else:
            st.dataframe(ev.head(20), hide_index=True, use_container_width=True)
    with right:
        st.subheader("Operational alerts")
        al = pd.DataFrame(st.session_state.get("live_alerts", []))
        if al.empty:
            st.success("No active simulated alerts.")
        else:
            st.dataframe(al.head(10), hide_index=True, use_container_width=True)


def render_live_tab(store: Dict[str, Any], bundle, settings: Dict[str, Any]) -> None:
    st.header("🎬 Live Demo")
    st.caption(
        "This demo automatically loads synthetic cross-platform advertising data, starts a simulated event stream, "
        "and generates a downloadable moving demo file. All values are labeled observed, estimated, or simulated."
    )

    ensure_live_state(store, bundle)

    if not st.session_state.get("live_seeded"):
        for _ in range(25):
            tick_live(store, bundle, settings.get("speed", "normal"))
        st.session_state.live_seeded = True

    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns(4)
    auto_live = ctrl1.checkbox("Auto live simulation", value=bool(settings.get("auto_live", True)), key="auto_live_checkbox")
    speed = ctrl2.select_slider("Simulation speed", options=["slow", "normal", "fast"], value=settings.get("speed", "normal"), key="speed_slider")
    if ctrl3.button("Reset live stream"):
        reset_live_state()
        st.rerun()
    if ctrl4.button("Regenerate demo video"):
        st.session_state.pop("demo_video", None)
        st.session_state.pop("demo_video_error", None)
        st.rerun()

    if "demo_video" not in st.session_state and "demo_video_error" not in st.session_state:
        try:
            with st.spinner("Generating live demo video..."):
                perf = store["performance"]
                live_attr = st.session_state.get("live_attr_revenue", {})
                if sum(float(v) for v in live_attr.values()) <= 0:
                    channel_rev = dict(zip(perf["platform"], perf["attributed_revenue_usd"]))
                else:
                    channel_rev = live_attr
                st.session_state.demo_video = generate_demo_video(
                    st.session_state.get("live_history", []),
                    channel_rev,
                    settings["model_display"],
                    store["markov_meta"].get("confidence", "unknown"),
                    store["data_quality"].get("score", 0.0),
                )
        except Exception as e:
            st.session_state.demo_video_error = str(e)

    vid_col, info_col = st.columns([2, 1])
    with vid_col:
        if "demo_video" in st.session_state:
            data, mime, fname = st.session_state.demo_video
            if mime == "video/mp4":
                st.video(data, format="video/mp4")
            elif mime == "image/gif":
                st.image(data, use_container_width=True)
            else:
                st.caption("Video generated. Use the download button below.")
            st.download_button("Download live demo video", data=data, file_name=fname, mime=mime)
        elif "demo_video_error" in st.session_state:
            st.warning(f"Video generation fallback failed: {st.session_state.demo_video_error}")
        else:
            st.info("Video generation pending...")

    with info_col:
        st.markdown(
            """
            **Demo interpretation**
            - Observed: simulated ad events and conversions.
            - Estimated: attribution credits from the selected model.
            - Synthetic: no real customer or advertiser data.
            - Governance: budget changes require human approval.
            """
        )

    try:
        if hasattr(st, "fragment"):
            @st.fragment(run_every=1.0)
            def live_fragment():
                try:
                    if st.session_state.get("auto_live_checkbox", True):
                        tick_live(store, bundle, st.session_state.get("speed_slider", "normal"))
                    render_live_outputs(store, bundle, settings)
                except Exception as ex:
                    st.error(f"Live fragment error: {ex}")

            live_fragment()
        else:
            raise AttributeError("st.fragment unavailable")
    except Exception:
        render_live_outputs(store, bundle, settings)
        b1, b2 = st.columns(2)
        if b1.button("Advance one tick"):
            tick_live(store, bundle, st.session_state.get("speed_slider", "normal"))
            st.rerun()
        if b2.button("Advance 10 ticks"):
            for _ in range(10):
                tick_live(store, bundle, st.session_state.get("speed_slider", "normal"))
            st.rerun()


# ============================================================================
# Workbench
# ============================================================================

def render_workbench(store: Dict[str, Any], bundle, settings: Dict[str, Any]) -> None:
    st.header("📊 Attribution Workbench")

    ds, ev_c, ev_q, co_c, co_q, jo_c, jo_q = bundle
    paths = store["paths"]
    credits_all = store["credits"]
    markov_meta = store["markov_meta"]
    importance = store["markov_importance"]

    c1, c2, c3 = st.columns([2, 2, 1])
    model_display = c1.selectbox(
        "Model",
        list(MODEL_MAP.keys()),
        index=list(MODEL_MAP.keys()).index(settings["model_display"]),
        key="workbench_model",
    )
    model_key = MODEL_MAP[model_display]
    selected_platforms = c2.multiselect("Platforms", PLATFORMS, default=PLATFORMS, key="workbench_platforms")
    conv_search = c3.text_input("Find conversion ID", key="conv_search")

    credits = credits_all[model_key].copy()
    if not credits.empty:
        credits = credits[credits["platform"].isin(selected_platforms)]

    last_shares = channel_credit_shares(credits_all["last_click"])
    perf = aggregate_performance(
        ev_c,
        co_c,
        credits_all[model_key],
        model_key,
        settings,
        markov_meta,
        last_shares,
    )
    dq = store["data_quality"]
    recs = make_recommendations(perf, markov_meta, dq, settings)

    total_rev = float(perf["attributed_revenue_usd"].sum())
    total_cost = float(perf["cost_usd"].sum())
    total_conv = float(perf["attributed_conversions"].sum())

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Attributed revenue", fmt_usd(total_rev))
    k2.metric("Cost", fmt_usd(total_cost))
    k3.metric("Blended ROAS", f"{safe_div(total_rev, total_cost):.2f}")
    k4.metric("Attributed conversions", f"{total_conv:,.1f}")
    k5.metric("Model confidence", markov_meta.get("confidence", "unknown"))

    if markov_meta.get("fallback_used"):
        st.warning(f"Deterministic fallback active: {markov_meta.get('fallback_reason', 'unknown reason')}")

    st.subheader("Channel performance")
    disp = perf.copy()
    disp["Cost"] = disp["cost_usd"].map(fmt_usd)
    disp["Attributed revenue"] = disp["attributed_revenue_usd"].map(fmt_usd)
    disp["Attributed margin"] = disp["attributed_margin_usd"].map(fmt_usd)
    disp["ROAS"] = disp["roas"].map(lambda x: f"{x:.2f}")
    disp["CAC"] = disp["cac_usd"].map(fmt_usd)
    disp["Share"] = disp["attributed_revenue_share"].map(lambda x: f"{x:.1%}")
    disp["Confidence"] = disp["confidence_score"].map(lambda x: f"{x:.2f}")
    st.dataframe(
        disp[
            [
                "platform",
                "Cost",
                "Attributed revenue",
                "Attributed margin",
                "ROAS",
                "CAC",
                "Share",
                "Confidence",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Budget recommendations")
    rec_disp = recs.copy()
    rec_disp["Estimated daily change"] = rec_disp["estimated_daily_budget_change_usd"].map(fmt_usd)
    rec_disp["Suggested change %"] = rec_disp["suggested_change_pct"].map(lambda x: f"{x:+.1f}%")
    st.dataframe(
        rec_disp[
            [
                "platform",
                "recommendation",
                "Suggested change %",
                "Estimated daily change",
                "requires_human_approval",
                "rationale",
                "risk_flags",
            ]
        ],
        hide_index=True,
        use_container_width=True,
    )
    st.caption("No external ad-platform write is performed. All spend changes are advisory and require human approval.")

    st.subheader("Daily attributed revenue trend")
    if paths.empty or credits.empty:
        st.info("No attributed paths available for the current filters.")
    else:
        trend = credits.merge(
            paths[["conversion_id", "conversion_time_utc"]],
            on="conversion_id",
            how="left",
        )
        trend["date"] = pd.to_datetime(trend["conversion_time_utc"]).dt.date
        trend["credited_revenue_usd"] = trend["credited_revenue_micros"].map(from_micros)
        trend_df = trend.groupby(["date", "platform"], as_index=False)["credited_revenue_usd"].sum()
        fig = px.line(
            trend_df,
            x="date",
            y="credited_revenue_usd",
            color="platform",
            title="Daily attributed revenue by platform",
        )
        fig.update_layout(height=340, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Conversion drill-down and evidence")
    if paths.empty:
        st.info("No conversions available.")
    else:
        candidate_ids = paths["conversion_id"].astype(str).tolist()
        if conv_search:
            matches = [cid for cid in candidate_ids if conv_search.lower() in cid.lower()]
            chosen = st.selectbox("Conversion", matches[:200] if matches else candidate_ids[:200])
        else:
            chosen = st.selectbox("Conversion", candidate_ids[:200])

        if chosen:
            prow = paths[paths["conversion_id"] == chosen]
            if not prow.empty:
                prow = prow.iloc[0]
                timeline = pd.DataFrame(
                    {
                        "sequence": range(1, len(prow["channels"]) + 1),
                        "platform": prow["channels"],
                        "lag_hours": [round(float(x), 2) for x in prow["lags_hours"]],
                        "event_type": prow["event_types"],
                    }
                )
                st.write("Touchpoint timeline")
                st.dataframe(timeline, hide_index=True, use_container_width=True)

                model_rows = []
                for mk, cr in credits_all.items():
                    sub = cr[cr["conversion_id"] == chosen]
                    if not sub.empty:
                        pivot = sub.groupby("platform")["credit_share"].sum()
                        for p, s in pivot.items():
                            model_rows.append({"model": mk, "platform": p, "credit_share": float(s)})
                if model_rows:
                    mr = pd.DataFrame(model_rows)
                    pivot = mr.pivot(index="platform", columns="model", values="credit_share").fillna(0.0)
                    st.write("Credit share by model for this conversion")
                    st.dataframe(pivot, use_container_width=True)

    st.subheader("Model evidence")
    ev1, ev2 = st.columns(2)
    with ev1:
        st.markdown("**Markov channel importance with bootstrap CI**")
        ci_rows = []
        for p in PLATFORMS:
            lo, hi = markov_meta.get("ci", {}).get(p, (0.0, 0.0))
            ci_rows.append(
                {
                    "platform": p,
                    "importance": float(importance.get(p, 0.0)),
                    "ci_lower": float(lo),
                    "ci_upper": float(hi),
                }
            )
        st.dataframe(pd.DataFrame(ci_rows), hide_index=True, use_container_width=True)
    with ev2:
        st.markdown("**Data quality status**")
        dq_rows = [
            {"metric": "Overall score", "value": f"{dq['score']:.1%}"},
            {"metric": "Consent coverage", "value": f"{dq['consent_coverage']:.1%}"},
            {"metric": "Event quarantine rate", "value": f"{dq['quarantine_rate']:.1%}"},
            {"metric": "Path coverage", "value": f"{dq['path_coverage']:.1%}"},
            {"metric": "Late arrival rate", "value": f"{dq['late_rate']:.1%}"},
            {"metric": "Fraud flag rate", "value": f"{dq['fraud_rate']:.1%}"},
            {"metric": "High revenue outlier rate", "value": f"{dq['outlier_rate']:.1%}"},
        ]
        st.dataframe(pd.DataFrame(dq_rows), hide_index=True, use_container_width=True)


# ============================================================================
# Testing, evaluation, and security
# ============================================================================

def _raw_event_row(**over: Any) -> Dict[str, Any]:
    now = utcnow()
    base = {
        "event_id": "e_test_1",
        "user_pseudo_id": "u_test_1",
        "session_id": "s_test_1",
        "timestamp_utc": now.isoformat(),
        "platform": "Google Search",
        "channel": "Google Search",
        "campaign": "camp_test",
        "ad_group": "ag_test",
        "creative_id": "cr_test",
        "device_type": "mobile",
        "geo": "US",
        "event_type": "click",
        "cost_micros": to_micros(1.0),
        "impression_count": 0,
        "click_count": 1,
        "consent_flag": True,
        "fraud_flag": False,
        "data_quality_flag": "",
        "source_system": "google_ads_api",
        "ingestion_time": now.isoformat(),
        "schema_version": SCHEMA_VERSION,
    }
    base.update(over)
    return base


def _raw_conversion_row(**over: Any) -> Dict[str, Any]:
    now = utcnow()
    base = {
        "conversion_id": "c_test_1",
        "user_pseudo_id": "u_test_1",
        "session_id": "s_test_1",
        "conversion_time_utc": now.isoformat(),
        "order_id": "o_test_1",
        "revenue_micros": to_micros(100.0),
        "margin_micros": to_micros(40.0),
        "product_category": "SaaS Trial",
        "new_customer_flag": True,
        "coupon_code": None,
        "consent_flag": True,
        "fraud_flag": False,
        "source_system": "internal_ets",
        "ingestion_time": now.isoformat(),
        "schema_version": SCHEMA_VERSION,
    }
    base.update(over)
    return base


def _paths_from_channels(channel_lists: List[List[str]], revenue_micros: int = to_micros(100.0)) -> pd.DataFrame:
    rows = []
    for i, chans in enumerate(channel_lists):
        if not chans:
            chans = ["Direct"]
        n = len(chans)
        rows.append(
            {
                "conversion_id": f"c_unit_{i}",
                "user_pseudo_id": f"u_unit_{i}",
                "conversion_time_utc": utcnow(),
                "revenue_micros": revenue_micros,
                "margin_micros": int(revenue_micros * 0.4),
                "channels": chans,
                "lags_hours": [24.0 * (n - j) for j in range(n)],
                "event_types": ["click"] * n,
                "has_click": True,
                "path_length": n,
            }
        )
    return pd.DataFrame(rows)


def run_self_tests() -> pd.DataFrame:
    results: List[Dict[str, str]] = []

    def check(name: str, fn) -> None:
        try:
            fn()
            results.append({"Test": name, "Status": "PASS", "Detail": ""})
        except AssertionError as e:
            results.append({"Test": name, "Status": "FAIL", "Detail": str(e)})
        except Exception as e:
            results.append({"Test": name, "Status": "ERROR", "Detail": f"{type(e).__name__}: {e}"})

    def test_micros_roundtrip():
        assert abs(from_micros(to_micros(1.234567)) - 1.234567) < 1e-9

    def test_safe_div():
        assert safe_div(10, 0) == 0.0
        assert abs(safe_div(10, 4) - 2.5) < 1e-9

    def test_validate_missing_platform():
        df = pd.DataFrame([_raw_event_row(platform=None)])
        clean, quar = validate_events(df)
        assert len(clean) == 0
        assert "missing_or_unknown_platform" in quar.iloc[0]["quarantine_reasons"]

    def test_validate_duplicate_event():
        row = _raw_event_row()
        df = pd.DataFrame([row, row])
        clean, quar = validate_events(df)
        assert len(clean) == 1
        assert len(quar) == 1
        assert "duplicate_event_id" in quar.iloc[0]["quarantine_reasons"]

    def test_validate_negative_cost():
        df = pd.DataFrame([_raw_event_row(cost_micros=-10)])
        clean, quar = validate_events(df)
        assert len(clean) == 0
        assert "negative_or_missing_cost" in quar.iloc[0]["quarantine_reasons"]

    def test_validate_future_timestamp():
        future = (utcnow() + timedelta(days=2)).isoformat()
        df = pd.DataFrame([_raw_event_row(timestamp_utc=future)])
        clean, quar = validate_events(df)
        assert len(clean) == 0
        assert "future_timestamp" in quar.iloc[0]["quarantine_reasons"]

    def test_validate_late_arrival_warning():
        now = utcnow()
        df = pd.DataFrame(
            [
                _raw_event_row(
                    timestamp_utc=now.isoformat(),
                    ingestion_time=(now + timedelta(days=3)).isoformat(),
                )
            ]
        )
        clean, quar = validate_events(df)
        assert len(clean) == 1
        assert len(quar) == 0
        assert "late_arriving" in clean.iloc[0]["data_quality_flag"]

    def test_validate_schema_mismatch():
        df = pd.DataFrame([_raw_event_row(schema_version="0.9.0")])
        clean, quar = validate_events(df)
        assert len(clean) == 0
        assert "schema_version_mismatch" in quar.iloc[0]["quarantine_reasons"]

    def test_validate_conversion_negative_revenue():
        df = pd.DataFrame([_raw_conversion_row(revenue_micros=-10)])
        clean, quar = validate_conversions(df)
        assert len(clean) == 0
        assert "negative_or_invalid_revenue" in quar.iloc[0]["quarantine_reasons"]

    def test_last_click_single_touch():
        paths = _paths_from_channels([["Meta Feed"]])
        cr = attribute_with_model(paths, "last_click")
        assert cr.iloc[0]["platform"] == "Meta Feed"
        assert abs(float(cr.iloc[0]["credit_share"]) - 1.0) < 1e-9

    def test_linear_sum_to_one():
        paths = _paths_from_channels([["Google Search", "Meta Feed", "Email"]])
        cr = attribute_with_model(paths, "linear")
        assert abs(float(cr["credit_share"].sum()) - 1.0) < 1e-9

    def test_time_decay_property():
        rng = np.random.default_rng(11)
        channel_lists = []
        for _ in range(25):
            n = int(rng.integers(1, 5))
            chans = rng.choice(ACTIONABLE_PLATFORMS, size=n, replace=True).tolist()
            channel_lists.append([str(c) for c in chans])
        paths = _paths_from_channels(channel_lists)
        for model in ["last_click", "first_click", "linear", "time_decay", "position"]:
            cr = attribute_with_model(paths, model)
            sums = cr.groupby("conversion_id")["credit_share"].sum()
            assert np.allclose(sums.values, 1.0, atol=1e-8)
            assert (cr["credit_share"] >= -1e-12).all()

    def test_markov_importance_sums_to_one():
        journeys = pd.DataFrame(
            [
                {"journey_id": "j1", "channels": ["Google Search", "Meta Feed"], "terminal": "conversion", "consent_flag": True},
                {"journey_id": "j2", "channels": ["Google Search"], "terminal": "null", "consent_flag": True},
                {"journey_id": "j3", "channels": ["Meta Feed", "Email"], "terminal": "conversion", "consent_flag": True},
                {"journey_id": "j4", "channels": ["Email"], "terminal": "null", "consent_flag": True},
            ]
        )
        imp, meta = compute_markov_importance(journeys, channels=PLATFORMS, n_bootstrap=3, min_paths=1, min_conv_prob=0.0)
        assert abs(sum(imp.values()) - 1.0) < 1e-8
        assert meta["n_journeys"] == 4

    def test_markov_fallback_low_data():
        journeys = pd.DataFrame(
            [
                {"journey_id": "j1", "channels": ["Google Search"], "terminal": "conversion", "consent_flag": True},
                {"journey_id": "j2", "channels": ["Meta Feed"], "terminal": "null", "consent_flag": True},
            ]
        )
        imp, meta = compute_markov_importance(journeys, channels=PLATFORMS, n_bootstrap=2, min_paths=300, min_conv_prob=0.01)
        assert meta["confidence"] == "low"
        assert meta["fallback_recommended"] is True

    def test_sql_roundtrip():
        conn = sqlite3.connect(":memory:")
        df = pd.DataFrame([_raw_event_row()])
        clean, quar = validate_events(df)
        _df_for_sql(clean).to_sql("events_clean", conn, if_exists="replace", index=False)
        out = pd.read_sql_query("SELECT * FROM events_clean", conn)
        assert len(out) == 1
        conn.close()

    def test_deterministic_generation():
        ref = datetime(2026, 1, 1, tzinfo=timezone.utc)
        a = _generate_dataset_impl(999, 40, ref)
        b = _generate_dataset_impl(999, 40, ref)
        pd.testing.assert_frame_equal(
            a["events"][["event_id", "platform", "cost_micros"]].sort_values("event_id").reset_index(drop=True),
            b["events"][["event_id", "platform", "cost_micros"]].sort_values("event_id").reset_index(drop=True),
        )

    def test_e2e_small_pipeline():
        ref = datetime(2026, 1, 1, tzinfo=timezone.utc)
        ds = _generate_dataset_impl(999, 120, ref)
        ev_c, ev_q = validate_events(ds["events"])
        co_c, co_q = validate_conversions(ds["conversions"])
        jo_c, jo_q = validate_journeys(ds["journeys"])
        paths = build_conversion_paths(ev_c, co_c, 7, True, True)
        assert not paths.empty
        cr = attribute_with_model(paths, "last_click")
        sums = cr.groupby("conversion_id")["credit_share"].sum()
        assert np.allclose(sums.values, 1.0, atol=1e-8)
        journeys_model = prepare_journeys_for_model(jo_c, co_c, True)
        imp, meta = compute_markov_importance(journeys_model, PLATFORMS, n_bootstrap=3, min_paths=10, min_conv_prob=0.0)
        mcr = markov_attribute_paths(paths, imp)
        msums = mcr.groupby("conversion_id")["credit_share"].sum()
        assert np.allclose(msums.values, 1.0, atol=1e-8)
        settings = {
            "window_days": 7,
            "include_view": True,
            "require_consent": True,
            "half_life_days": 2.0,
            "model": "markov",
            "target_roas": 2.0,
            "min_confidence": "medium",
            "force_fallback": False,
            "safe_mode": False,
        }
        perf = aggregate_performance(ev_c, co_c, mcr, "markov", settings, meta, channel_credit_shares(cr))
        assert not perf.empty
        assert perf["cost_usd"].min() >= 0.0

    def test_decision_approval_guardrail():
        perf = pd.DataFrame(
            [
                {
                    "platform": "Meta Feed",
                    "cost_usd": 1000.0,
                    "attributed_revenue_usd": 3000.0,
                    "attributed_margin_usd": 1200.0,
                    "attributed_conversions": 30.0,
                    "roas": 3.0,
                    "margin_roas": 1.2,
                    "cac_usd": 33.3,
                    "attributed_revenue_share": 0.5,
                    "disagreement_vs_last_click": 0.05,
                    "confidence_score": 0.9,
                    "model": "markov",
                }
            ]
        )
        meta = {"confidence": "high", "fallback_used": False, "ci": {"Meta Feed": (0.45, 0.55)}}
        dq = {"consent_coverage": 0.95, "quarantine_rate": 0.01, "fraud_rate": 0.005}
        settings = {"target_roas": 2.0, "min_confidence": "medium", "safe_mode": False}
        recs = make_recommendations(perf, meta, dq, settings)
        assert recs.iloc[0]["recommendation"] == "Increase"
        assert bool(recs.iloc[0]["requires_human_approval"]) is True

        settings_safe = dict(settings)
        settings_safe["safe_mode"] = True
        recs_safe = make_recommendations(perf, meta, dq, settings_safe)
        assert recs_safe.iloc[0]["recommendation"] == "Hold"

    check("Micros roundtrip", test_micros_roundtrip)
    check("Safe division", test_safe_div)
    check("Validation: missing platform", test_validate_missing_platform)
    check("Validation: duplicate event ID", test_validate_duplicate_event)
    check("Validation: negative cost", test_validate_negative_cost)
    check("Validation: future timestamp", test_validate_future_timestamp)
    check("Validation: late arrival warning", test_validate_late_arrival_warning)
    check("Validation: schema mismatch", test_validate_schema_mismatch)
    check("Validation: negative conversion revenue", test_validate_conversion_negative_revenue)
    check("Attribution: last click single touch", test_last_click_single_touch)
    check("Attribution: linear sums to one", test_linear_sum_to_one)
    check("Attribution: property test all rule models", test_time_decay_property)
    check("Markov: importance sums to one", test_markov_importance_sums_to_one)
    check("Markov: low-data fallback", test_markov_fallback_low_data)
    check("Persistence: SQLite roundtrip", test_sql_roundtrip)
    check("Reproducibility: deterministic generation", test_deterministic_generation)
    check("E2E: small pipeline", test_e2e_small_pipeline)
    check("Governance: decision approval guardrail", test_decision_approval_guardrail)

    return pd.DataFrame(results)


def run_controlled_evaluation(store: Dict[str, Any], bundle, settings: Dict[str, Any]):
    ds, ev_c, ev_q, co_c, co_q, jo_c, jo_q = bundle
    paths = store["paths"]
    truth = ds["truth"]

    journeys = prepare_journeys_for_model(jo_c, co_c, settings["require_consent"]).copy()
    if journeys.empty:
        return None

    journeys["_sort_time"] = journeys["conversion_time"].fillna(pd.Timestamp("2000-01-01", tz="UTC"))
    journeys = journeys.sort_values("_sort_time").reset_index(drop=True)

    n = len(journeys)
    i_train = int(0.70 * n)
    i_val = int(0.85 * n)
    train = journeys.iloc[:i_train]
    test = journeys.iloc[i_val:]

    test_conv_ids = set(test[test["terminal"] == "conversion"]["conversion_id"].dropna().astype(str))
    test_paths = paths[paths["conversion_id"].astype(str).isin(test_conv_ids)]
    if test_paths.empty:
        return None

    imp_train, meta_train = compute_markov_importance(
        train,
        channels=PLATFORMS,
        n_bootstrap=10,
        seed=SEED + 17,
        min_paths=50,
        min_conv_prob=0.005,
        require_consent=settings["require_consent"],
    )

    if meta_train.get("confidence") == "low":
        markov_credits = attribute_with_model(test_paths, "linear", settings["half_life_days"])
        model_used = "linear_fallback"
    else:
        markov_credits = markov_attribute_paths(test_paths, imp_train)
        model_used = "markov"

    last_credits = attribute_with_model(test_paths, "last_click")

    est_m = channel_credit_shares(markov_credits)
    est_l = channel_credit_shares(last_credits)
    true_s = pd.Series({p: float(truth.get(p, 0.0)) for p in PLATFORMS})
    true_s = true_s / max(1e-12, float(true_s.sum()))

    mae_m = float(np.mean(np.abs(est_m - true_s)))
    rmse_m = float(np.sqrt(np.mean((est_m - true_s) ** 2)))
    mae_l = float(np.mean(np.abs(est_l - true_s)))
    rmse_l = float(np.sqrt(np.mean((est_l - true_s) ** 2)))

    try:
        spearman_m = float(est_m.corr(true_s, method="spearman"))
    except Exception:
        spearman_m = float("nan")

    conv_rev = markov_credits.pivot_table(
        index="conversion_id",
        columns="platform",
        values="credited_revenue_micros",
        aggfunc="sum",
    ).reindex(columns=PLATFORMS, fill_value=0.0)
    ids = conv_rev.index.to_numpy()
    boot_shares = []
    if len(ids) >= 5:
        rng = np.random.default_rng(SEED + 29)
        for _ in range(50):
            sample_idx = rng.integers(0, len(ids), len(ids))
            samp = conv_rev.iloc[sample_idx]
            tot = float(samp.sum().sum())
            if tot > 0:
                boot_shares.append((samp.sum() / tot).reindex(PLATFORMS).fillna(0.0).to_numpy())
    if boot_shares:
        arr = np.vstack(boot_shares)
        lower = np.percentile(arr, 2.5, axis=0)
        upper = np.percentile(arr, 97.5, axis=0)
    else:
        lower = est_m.to_numpy()
        upper = est_m.to_numpy()

    table = pd.DataFrame(
        {
            "platform": PLATFORMS,
            "true_share": [float(true_s[p]) for p in PLATFORMS],
            "markov_share": [float(est_m[p]) for p in PLATFORMS],
            "last_click_share": [float(est_l[p]) for p in PLATFORMS],
            "abs_error": [abs(float(est_m[p]) - float(true_s[p])) for p in PLATFORMS],
            "ci_lower": [float(lower[i]) for i in range(len(PLATFORMS))],
            "ci_upper": [float(upper[i]) for i in range(len(PLATFORMS))],
        }
    )

    regret_proxy = float(0.5 * np.abs(est_m - true_s).sum())

    metrics = {
        "model_used": model_used,
        "train_journeys": len(train),
        "test_journeys": len(test),
        "test_conversions": len(test_paths),
        "markov_mae": mae_m,
        "markov_rmse": rmse_m,
        "last_click_mae": mae_l,
        "last_click_rmse": rmse_l,
        "spearman": spearman_m,
        "regret_proxy": regret_proxy,
        "confidence": meta_train.get("confidence"),
    }

    fig = go.Figure()
    x = np.arange(len(PLATFORMS))
    width = 0.25
    fig.add_trace(go.Bar(x=x - width, y=table["true_share"], name="Synthetic ground truth"))
    fig.add_trace(go.Bar(x=x, y=table["markov_share"], name="Markov / fallback"))
    fig.add_trace(go.Bar(x=x + width, y=table["last_click_share"], name="Last click"))
    fig.update_layout(
        title="Controlled evaluation: estimated vs synthetic ground-truth channel shares",
        xaxis=dict(tickangle=25, tickmode="array", tickvals=x, ticktext=PLATFORMS),
        yaxis_title="Revenue share",
        barmode="group",
        height=380,
        margin=dict(l=20, r=20, t=60, b=80),
    )
    return table, metrics, fig


# ============================================================================
# Product artifacts
# ============================================================================

def user_journeys_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Journey": "Growth marketer morning budget review",
                "User": "Performance marketing manager",
                "Legacy workflow": "Open Google, Meta, TikTok, email dashboards; copy numbers into spreadsheet; argue about last-click vs platform-reported conversions.",
                "Improved decision": "See cross-channel credited revenue, confidence, disagreement with last-click, and capped budget recommendations.",
                "Failure case": "View-through conversions inflate TikTok credit; model confidence low.",
                "Acceptance check": "Dashboard shows fallback reason and suppresses auto-increase; human approval required for any change.",
            },
            {
                "Journey": "Revenue operations monthly close",
                "User": "RevOps analyst",
                "Legacy workflow": "Reconcile finance revenue, ad-platform revenue, coupons, refunds, and currency manually.",
                "Improved decision": "Audit attributed revenue against validated conversions and quarantine records.",
                "Failure case": "Late-arriving conversions after close window; duplicate conversion IDs.",
                "Acceptance check": "Quarantine log explains duplicates/late records; attributed revenue sums to validated conversion revenue within tolerance.",
            },
            {
                "Journey": "Marketing data engineer pipeline monitoring",
                "User": "Marketing data engineer",
                "Legacy workflow": "Cron jobs dump CSVs; failures discovered by email or missing reports.",
                "Improved decision": "Schema validation, data-quality score, replayable ingestion, and audit history.",
                "Failure case": "TikTok API changes field name; schema_version mismatch; negative cost spikes.",
                "Acceptance check": "Malformed records quarantined, dashboard alerts, deterministic fallback prevents bad budget advice.",
            },
        ]
    )


def functional_requirements_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ID": "FR-01", "Requirement": "Ingest synthetic event and conversion records from multiple named source adapters.", "Acceptance": "Raw event count > 0; source_system field populated; demo loads without API keys."},
            {"ID": "FR-02", "Requirement": "Validate schemas, identifiers, timestamps, consent flags, costs, and revenue units.", "Acceptance": "Blocking issues quarantined; warnings flagged; no silent coercion of negative cost/revenue."},
            {"ID": "FR-03", "Requirement": "Build canonical user journeys and conversion attribution paths within configurable lookback windows.", "Acceptance": "Paths exclude post-conversion touches; respect consent policy; Direct fallback when no valid touch."},
            {"ID": "FR-04", "Requirement": "Compute transparent baselines: last-click, first-click, linear, time-decay, position-based.", "Acceptance": "Credits sum to 1 per conversion; deterministic outputs for fixed seed."},
            {"ID": "FR-05", "Requirement": "Estimate data-driven channel credit using Markov removal effects with uncertainty.", "Acceptance": "Bootstrap CI reported; low-confidence cases trigger fallback."},
            {"ID": "FR-06", "Requirement": "Score channels by ROAS, margin ROAS, CAC, confidence, disagreement, and risk flags.", "Acceptance": "No recommendation for Direct or zero-cost channels; risk flags visible."},
            {"ID": "FR-07", "Requirement": "Produce human-approved budget recommendations with caps and safe mode.", "Acceptance": "Any non-zero spend change requires approval; safe mode forces Hold."},
            {"ID": "FR-08", "Requirement": "Persist metadata, model runs, quarantine records, and audit history in SQLite.", "Acceptance": "Audit table contains model_run, test_suite_run, evaluation_run, approval events."},
            {"ID": "FR-09", "Requirement": "Provide dashboard with live simulation, filters, drill-downs, explanations, and demo video generation.", "Acceptance": "App auto-loads data, starts simulation, and produces downloadable moving demo file."},
            {"ID": "FR-10", "Requirement": "Run local tests and controlled evaluation without paid services.", "Acceptance": "Test suite passes; evaluation artifact displays metrics and CI."},
        ]
    )


def nonfunctional_requirements_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ID": "NFR-01", "Category": "Portability", "Requirement": "Run on Windows 10/11, CPU-only, Intel integrated graphics acceptable.", "Acceptance": "streamlit run app.py works after pip install -r requirements.txt."},
            {"ID": "NFR-02", "Category": "Cost", "Requirement": "No paid APIs or proprietary cloud services for core demo.", "Acceptance": "No API keys required; all data synthetic/local."},
            {"ID": "NFR-03", "Category": "Reproducibility", "Requirement": "Deterministic synthetic generation and model runs for fixed seed/settings.", "Acceptance": "Deterministic test passes; settings hash stored in audit."},
            {"ID": "NFR-04", "Category": "Latency", "Requirement": "Interactive dashboard operations should feel responsive on 16 GB RAM CPU laptop.", "Acceptance": "Default 9k journeys attribution run completes in seconds; live tick lightweight."},
            {"ID": "NFR-05", "Category": "Governance", "Requirement": "Consequential actions require human approval and audit logging.", "Acceptance": "Approval UI logs action; no external write path exists."},
            {"ID": "NFR-06", "Category": "Privacy", "Requirement": "Use only synthetic or appropriately licensed data; no real PII.", "Acceptance": "Data provenance documented; pseudo IDs hashed."},
            {"ID": "NFR-07", "Category": "Reliability", "Requirement": "Graceful degradation when model quality or input quality is inadequate.", "Acceptance": "Fallback to linear/hold with visible reason."},
            {"ID": "NFR-08", "Category": "Maintainability", "Requirement": "Single-file prototype now, modular boundaries documented for production split.", "Acceptance": "Architecture and folder plan present; module functions separated logically."},
        ]
    )


def prototype_vs_production_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Dimension": "Data", "Useful prototype": "Synthetic/local CSV/DataFrame ingestion, SQLite.", "Production enterprise service": "Managed event lake, Kafka/Pulsar, dbt, Data Quality SLAs, PII governance."},
            {"Dimension": "Identity", "Prototype": "Pseudo user IDs, simple consent flag.", "Production": "Privacy-safe identity resolution, consent ledger, regional residency controls."},
            {"Dimension": "Model", "Prototype": "Rule baselines + Markov removal effect with bootstrap CI.", "Production": "Causal incrementality, geo/PSA holdouts, hierarchical Bayesian or ML credit models, drift monitoring."},
            {"Dimension": "API", "Prototype": "Streamlit functions and local service boundary.", "Production": "FastAPI/gRPC, authN/authZ, rate limits, tenant isolation, contract tests."},
            {"Dimension": "Operations", "Prototype": "Local logs, SQLite audit, manual restart.", "Production": "Kubernetes, observability, on-call, SLOs, backup/DR, rollback, change management."},
            {"Dimension": "Decisions", "Prototype": "Advisory recommendations with human approval.", "Production": "Guarded automation, budget caps, approval workflows, finance integration, audit evidence."},
        ]
    )


def risk_register_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ID": "R1", "Risk": "Data availability/outage", "Likelihood": "Medium", "Impact": "High", "Mitigation": "Idempotent batches, quarantine, late-arrival handling, coverage guardrails.", "Detection": "Data-quality score, ingestion success rate, quarantine alerts.", "Human approval": "Source onboarding/offboarding."},
            {"ID": "R2", "Risk": "Model error / misattribution", "Likelihood": "High", "Impact": "High", "Mitigation": "Baseline comparison, bootstrap CI, fallback, segment checks.", "Detection": "Disagreement with last-click, low confidence, evaluation MAE.", "Human approval": "Model promotion."},
            {"ID": "R3", "Risk": "Privacy / re-identification", "Likelihood": "Low", "Impact": "Critical", "Mitigation": "Synthetic data, hashed pseudo IDs, no real PII, retention limits.", "Detection": "Schema scans, audit logs, privacy review.", "Human approval": "Data retention change, real-data pilot."},
            {"ID": "R4", "Risk": "Fairness / segment bias", "Likelihood": "Medium", "Impact": "High", "Mitigation": "Geo/device/new-customer segment reports; avoid protected attributes.", "Detection": "Subgroup performance drift.", "Human approval": "Segment exclusion policy."},
            {"ID": "R5", "Risk": "Misuse / unsafe automation", "Likelihood": "Medium", "Impact": "Critical", "Mitigation": "No external writes, caps, safe mode, approval required.", "Detection": "Approval log, recommendation audit.", "Human approval": "Spend change."},
            {"ID": "R6", "Risk": "Operational failure", "Likelihood": "Medium", "Impact": "Medium", "Mitigation": "Retries, idempotency, backpressure, graceful degradation.", "Detection": "Error rate, latency, restart recovery tests.", "Human approval": "Infrastructure scaling/cost change."},
            {"ID": "R7", "Risk": "Adoption resistance", "Likelihood": "High", "Impact": "Medium", "Mitigation": "Explanations, evidence view, pilot with marketer oversight.", "Detection": "Usage metrics, feedback, override rate.", "Human approval": "Policy change."},
            {"ID": "R8", "Risk": "Cost overrun", "Likelihood": "Medium", "Impact": "Medium", "Mitigation": "CPU-first, caching, sampling, modular monolith.", "Detection": "Cost per operation, CPU/memory metrics.", "Human approval": "Paid compute enablement."},
        ]
    )


def backlog_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ID": "BL-01", "Priority": "P0", "Story": "As a marketer, I want validated synthetic cross-platform touchpoints so I can trust the demo data.", "Acceptance": "Ingestion and validation pass; quarantine visible."},
            {"ID": "BL-02", "Priority": "P0", "Story": "As an analyst, I want last-click and linear baselines so I can compare advanced attribution.", "Acceptance": "Credits sum to one; deterministic."},
            {"ID": "BL-03", "Priority": "P0", "Story": "As a product owner, I want a live dashboard with auto-loaded demo data so stakeholders can evaluate value quickly.", "Acceptance": "App starts with non-blank dashboard and simulation."},
            {"ID": "BL-04", "Priority": "P1", "Story": "As a data scientist, I want Markov removal-effect scoring with uncertainty so credit is data-driven and auditable.", "Acceptance": "Bootstrap CI and fallback behavior shown."},
            {"ID": "BL-05", "Priority": "P1", "Story": "As a finance partner, I want attributed revenue reconciled to validated conversions.", "Acceptance": "Revenue sums match within tolerance; audit log available."},
            {"ID": "BL-06", "Priority": "P1", "Story": "As a governance lead, I want human approval for spend changes and safe mode.", "Acceptance": "Approval required; safe mode forces Hold."},
            {"ID": "BL-07", "Priority": "P2", "Story": "As a platform engineer, I want Docker Compose and API contracts for future productionization.", "Acceptance": "Docs include optional deployment path."},
            {"ID": "BL-08", "Priority": "P2", "Story": "As a causal inference researcher, I want geo/PSA holdout integration to validate incrementality.", "Acceptance": "Out of scope for prototype; roadmap documented."},
        ]
    )


def glossary_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Term": "Touchpoint", "Definition": "An observable ad interaction such as impression, click, view, video view, or email open."},
            {"Term": "Conversion path", "Definition": "Ordered set of pre-conversion touchpoints within an attribution window for a user."},
            {"Term": "Attribution credit", "Definition": "Fraction of conversion revenue assigned to a platform or touchpoint."},
            {"Term": "Last click", "Definition": "Baseline model assigning all credit to the final click before conversion."},
            {"Term": "Markov removal effect", "Definition": "Data-driven credit based on reduction in conversion probability when a channel is removed."},
            {"Term": "Consent coverage", "Definition": "Share of events/conversions marked as consented and eligible for modeling."},
            {"Term": "Quarantine", "Definition": "Records blocked from canonical modeling due to schema or quality failures."},
            {"Term": "Fallback", "Definition": "Deterministic simpler model used when advanced model confidence or input quality is inadequate."},
            {"Term": "ROAS", "Definition": "Return on ad spend, attributed revenue divided by cost."},
            {"Term": "CAC", "Definition": "Customer acquisition cost, cost divided by attributed conversions."},
        ]
    )


def render_page1() -> None:
    st.header("📄 Page 1 — Product Requirements, Users, Value, and Risk")

    st.subheader("Precise product mission")
    st.markdown(
        """
        **Mission:** Provide a governed, reproducible cross-platform attribution engine that converts fragmented
        advertising touchpoints into auditable revenue credit, channel performance estimates, uncertainty-aware
        budget recommendations, and human-approved optimization actions.

        **Measurable scope for this prototype:**
        - 8 synthetic platforms: Google Search, Meta Feed, TikTok Video, YouTube, Email, Display Retargeting, Affiliate, Direct.
        - 90 days of synthetic journeys, roughly 9,000 journeys and tens of thousands of events.
        - CPU-only Windows-compatible Streamlit app.
        - Local SQLite persistence for canonical records, quarantine, and audit history.
        - Rule-based baselines plus Markov removal-effect attribution with bootstrap confidence intervals.
        - Live simulated event stream and downloadable moving demo video.
        - No paid APIs, no real customer data, no external ad-platform writes.
        """
    )

    st.subheader("Real operational problem and why a large technology company would pay")
    st.markdown(
        """
        Large advertisers and ad-tech platforms lose money when budget decisions are based on inconsistent,
        platform-siloed attribution. Each platform claims conversions, finance uses a different revenue definition,
        and marketers reconcile spreadsheets manually. The operational problem is not only “which channel gets credit?”
        but also “can we trust the data, explain the decision, and act safely?”

        A large technology company would pay to solve this because:
        - Misallocated ad spend directly reduces contribution margin.
        - Attribution disputes slow budget velocity and create political decision-making.
        - Data-quality failures can cause costly over- or under-investment.
        - Governance requirements demand auditability, consent handling, and human oversight.
        - Reliable cross-channel insight improves forecasting, creative testing, and bid/budget strategy.
        """
    )

    st.subheader("Who uses it and what decisions it improves")
    st.markdown(
        """
        - **Performance marketers:** channel budget shifts, creative/platform testing priorities.
        - **Revenue operations / finance:** reconciled attributed revenue, CAC validation, close audit trail.
        - **Marketing data engineers:** pipeline health, schema drift, quarantine replay decisions.
        - **Data scientists / ML engineers:** model comparison, uncertainty, fallback behavior.
        - **Governance / legal / privacy reviewers:** consent coverage, data retention, approval logs.
        """
    )

    st.subheader("Three realistic user journeys")
    st.dataframe(user_journeys_df(), hide_index=True, use_container_width=True)

    st.subheader("Functional requirements")
    st.dataframe(functional_requirements_df(), hide_index=True, use_container_width=True)

    st.subheader("Non-functional requirements")
    st.dataframe(nonfunctional_requirements_df(), hide_index=True, use_container_width=True)

    st.subheader("Assumptions, dependencies, out-of-scope")
    st.markdown(
        """
        **Assumptions**
        - Synthetic data is representative enough for demo and design validation.
        - Revenue is modeled in micro-USD integers to avoid floating-point accounting errors.
        - Consent flags are available at event and conversion level.
        - Attribution windows are configurable but default to 7 days.

        **Dependencies**
        - Python 3.10+.
        - Open-source packages: Streamlit, pandas, NumPy, Plotly, Pillow, OpenCV headless.
        - Local filesystem access for SQLite demo database.

        **Out of scope for initial prototype**
        - Real ad-platform API connections.
        - Production identity resolution across authenticated users.
        - Automated bid/budget execution.
        - Causal geo experiments.
        - Multi-tenant SaaS authentication and billing.
        - Paid cloud services or proprietary APIs.
        """
    )

    st.subheader("Release milestones")
    st.markdown(
        """
        - **M0:** Environment, configuration, license inventory.
        - **M1:** Synthetic/open data preparation and provenance.
        - **M2:** Schema validation, quarantine, canonical storage.
        - **M3:** Baseline analytics and rule-based attribution.
        - **M4:** Markov removal-effect model, uncertainty, fallback.
        - **M5:** API/service boundary and Streamlit dashboard.
        - **M6:** Audit trail, tests, controlled evaluation.
        - **M7:** Packaging, demo video, docs, acceptance.
        """
    )

    st.subheader("Useful prototype vs production enterprise service")
    st.dataframe(prototype_vs_production_df(), hide_index=True, use_container_width=True)

    st.subheader("Measurable value hypothesis")
    st.markdown(
        """
        **Hypothesis:** If marketers replace fragmented last-click spreadsheets with auditable cross-platform
        attribution and guarded recommendations, then wasted spend decreases and budget decision latency improves,
        provided data coverage and model confidence are sufficient.

        **Outcome metrics**
        - Reduction in blended CAC or increase in contribution-margin ROAS under controlled tests.
        - Reduction in disputed channel revenue during monthly close.
        - Faster budget decision cycle time.

        **Leading indicators**
        - Consent coverage, path coverage, quarantine rate.
        - Model disagreement with last-click.
        - Recommendation adoption rate.
        - Time from data ingestion to approved decision.

        **Quality metrics**
        - MAE/RMSE against synthetic ground truth.
        - Spearman rank correlation.
        - Bootstrap CI width and calibration.
        - Segment-level error by geo/device/new customer.

        **Reliability metrics**
        - Ingestion success rate.
        - p95 processing latency.
        - Error rate and restart recovery.
        - Audit completeness.

        **Guardrails**
        - No spend change without human approval.
        - Low confidence forces Hold/Investigate.
        - Data quality below threshold triggers fallback.
        - Synthetic/demo values clearly labeled.
        """
    )

    st.subheader("Baseline and controlled evaluation plan")
    st.markdown(
        """
        **Baseline:** current manual last-click spreadsheet and platform-reported conversions.

        **Controlled evaluation:**
        1. Use synthetic ground-truth incremental shares to test whether attribution recovers known structure.
        2. Use chronological train/validation/test splits to avoid future leakage.
        3. Compare Markov/fallback to last-click using MAE, RMSE, rank correlation, and regret proxy.
        4. For real deployment, run geo or PSA holdout experiments before claiming incremental revenue impact.
        """
    )

    st.subheader("Risk register")
    st.dataframe(risk_register_df(), hide_index=True, use_container_width=True)

    st.subheader("Prioritized backlog")
    st.dataframe(backlog_df(), hide_index=True, use_container_width=True)

    st.subheader("Glossary")
    st.dataframe(glossary_df(), hide_index=True, use_container_width=True)

    st.subheader("Executive summary")
    st.markdown(
        """
        This project delivers a reviewable, CPU-first prototype of a cross-platform attribution engine for advertising
        revenue optimization. It is not a generic AI chatbot. It is a governed decision-support system that turns
        fragmented ad touchpoints into auditable credit, uncertainty-aware channel scores, and human-approved
        budget recommendations. The prototype includes synthetic data generation, schema validation, canonical storage,
        baseline and Markov attribution, live simulation, dashboard explanations, tests, evaluation, risk controls,
        and a downloadable moving demo.
        """
    )

    st.subheader("Page 1 acceptance checks")
    st.markdown(
        """
        - [ ] Problem, users, decisions, and legacy workflow are concrete.
        - [ ] At least three user journeys include failure cases and acceptance checks.
        - [ ] Functional/non-functional requirements are project-specific.
        - [ ] Prototype vs production distinction is explicit.
        - [ ] Value hypothesis includes outcome, leading, quality, reliability metrics, and guardrails.
        - [ ] Risk register covers data, model, privacy, fairness, misuse, operations, adoption, and cost.
        - [ ] Human-approval actions are identified.
        - [ ] Synthetic/licensed data requirement is satisfied.
        """
    )


def render_page2() -> None:
    st.header("🧱 Page 2 — Architecture, Data Contracts, and Algorithm Design")

    st.subheader("Modular architecture")
    st.markdown(
        """
        The prototype is a modular monolith in one file for portability, but boundaries are explicit:
        source adapters → ingestion/validation → canonical model → path/feature processing → baseline analytics →
        model layer → decision scoring → API/dashboard → audit/observability.
        """
    )

    st.markdown(
        """
        ```mermaid
        flowchart LR
          subgraph Sources
            GA[Google Ads adapter]
            MA[Meta adapter]
            TA[TikTok adapter]
            EA[Email adapter]
            IA[Internal ETS adapter]
          end
          GA-->ING[Ingestion + schema validation]
          MA-->ING
          TA-->ING
          EA-->ING
          IA-->ING
          ING-->Q[(Quarantine)]
          ING-->CAN[(Canonical SQLite)]
          CAN-->PATH[Path builder]
          PATH-->RULE[Rule baselines]
          PATH-->MK[Markov trainer]
          RULE-->SCORE[Decision scorer]
          MK-->SCORE
          SCORE-->REC[Budget recommendations]
          REC-->DASH[Streamlit dashboard]
          SCORE-->AUDIT[(Audit history)]
          ING-->OBS[Observability / data quality]
        ```
        """
    )

    st.markdown(
        """
        ```mermaid
        sequenceDiagram
          participant Marketer
          participant Dashboard
          participant Engine
          participant Validator
          participant Model
          participant AuditDB
          Marketer->>Dashboard: Select window/model/filters
          Dashboard->>Engine: Run attribution
          Engine->>Validator: Validate canonical records
          Validator-->>Engine: Clean paths + quality flags
          Engine->>Model: Train Markov / compute baselines
          Model-->>Engine: Credits + CI + fallback status
          Engine->>AuditDB: Write model run evidence
          Engine-->>Dashboard: Channel scores + recommendations
          Dashboard-->>Marketer: Explain result and request approval
        ```
        """
    )

    ownership = pd.DataFrame(
        [
            {"Component": "Source adapters", "Owner": "Marketing data engineering", "Failure boundary": "API outage or schema drift", "Mitigation": "Idempotent ingestion, quarantine, replay"},
            {"Component": "Ingestion/validation", "Owner": "Data platform", "Failure boundary": "Malformed records", "Mitigation": "Blocking vs warning issues, data-quality score"},
            {"Component": "Canonical model", "Owner": "Analytics engineering", "Failure boundary": "Duplicate IDs, late arrivals", "Mitigation": "Deduplication, event-time vs ingestion-time"},
            {"Component": "Path builder", "Owner": "Attribution service", "Failure boundary": "Identity or window errors", "Mitigation": "Consent filter, Direct fallback, configurable window"},
            {"Component": "Model layer", "Owner": "Applied AI", "Failure boundary": "Sparse data, low confidence", "Mitigation": "Bootstrap CI, deterministic fallback"},
            {"Component": "Decision scorer", "Owner": "Product + finance", "Failure boundary": "Unsafe automation", "Mitigation": "Caps, approval, safe mode"},
            {"Component": "Dashboard/API", "Owner": "Product engineering", "Failure boundary": "UX mistrust", "Mitigation": "Evidence view, labels, audit history"},
            {"Component": "Audit/observability", "Owner": "Governance/SRE", "Failure boundary": "Unexplainable decisions", "Mitigation": "Immutable logs, retention policy"},
        ]
    )
    st.dataframe(ownership, hide_index=True, use_container_width=True)

    st.subheader("Data contracts")
    contract = pd.DataFrame(
        [
            {"field": "event_id", "type": "string", "unit": "identifier", "required": True, "notes": "Globally unique; duplicates quarantined."},
            {"field": "user_pseudo_id", "type": "string", "unit": "hashed identifier", "required": True, "notes": "No real PII; synthetic hash."},
            {"field": "timestamp_utc", "type": "datetime", "unit": "UTC ISO-8601", "required": True, "notes": "Event time; future records blocked."},
            {"field": "platform", "type": "enum", "unit": "channel", "required": True, "notes": "Must be in PLATFORMS."},
            {"field": "event_type", "type": "enum", "unit": "interaction", "required": True, "notes": "click, view, video_view, impression, email_open."},
            {"field": "cost_micros", "type": "integer", "unit": "micro-USD", "required": True, "notes": "Non-negative; outliers flagged."},
            {"field": "consent_flag", "type": "boolean", "unit": "policy", "required": True, "notes": "Missing defaults false and warns."},
            {"field": "schema_version", "type": "string", "unit": "contract", "required": True, "notes": "Mismatch quarantined."},
            {"field": "conversion_id", "type": "string", "unit": "identifier", "required": True, "notes": "Unique conversion; duplicates quarantined."},
            {"field": "revenue_micros", "type": "integer", "unit": "micro-USD", "required": True, "notes": "Non-negative; high outliers flagged."},
            {"field": "margin_micros", "type": "integer", "unit": "micro-USD", "required": False, "notes": "Imputed with warning if missing."},
            {"field": "ingestion_time", "type": "datetime", "unit": "UTC ISO-8601", "required": True, "notes": "Used for late-arrival detection."},
        ]
    )
    st.dataframe(contract, hide_index=True, use_container_width=True)

    st.markdown("**Example canonical event record**")
    st.code(
        json_dumps(
            {
                "event_id": "e_9f2a...",
                "user_pseudo_id": "u_5c1d...",
                "session_id": "s_77ab...",
                "timestamp_utc": "2026-10-29T14:22:10Z",
                "platform": "TikTok Video",
                "event_type": "video_view",
                "cost_micros": 38000,
                "consent_flag": True,
                "schema_version": "1.0.0",
                "data_quality_flag": "late_arriving",
            },
            indent=2,
        ),
        language="json",
    )

    st.markdown("**Example canonical conversion record**")
    st.code(
        json_dumps(
            {
                "conversion_id": "c_18be...",
                "user_pseudo_id": "u_5c1d...",
                "conversion_time_utc": "2026-10-30T09:05:00Z",
                "order_id": "o_44d1...",
                "revenue_micros": 9200000,
                "margin_micros": 3680000,
                "product_category": "App Subscription",
                "consent_flag": True,
                "schema_version": "1.0.0",
            },
            indent=2,
        ),
        language="json",
    )

    handling = pd.DataFrame(
        [
            {"case": "Missing platform", "handling": "Block and quarantine; cannot attribute."},
            {"case": "Duplicate event_id", "handling": "Keep first by ingestion order; quarantine later duplicates."},
            {"case": "Late-arriving record", "handling": "Accept if within retention; flag late_arriving; exclude from real-time alerts if too late."},
            {"case": "Malformed timestamp", "handling": "Block; quarantine with bad_timestamp."},
            {"case": "Negative cost/revenue", "handling": "Block; quarantine; never silently coerce to zero."},
            {"case": "Outlier revenue", "handling": "Accept with high_revenue_outlier warning; segment review."},
            {"case": "Schema change", "handling": "Quarantine unknown schema_version; require adapter version upgrade."},
            {"case": "Missing consent", "handling": "Default false, warn, exclude from model if require_consent=true."},
        ]
    )
    st.dataframe(handling, hide_index=True, use_container_width=True)

    st.subheader("Transparent baseline before advanced ML")
    approaches = pd.DataFrame(
        [
            {
                "Approach": "Last click",
                "Role": "Transparent baseline",
                "Pros": "Simple, explainable, matches some platform reports.",
                "Cons": "Ignores assist channels, biased toward closed-loop platforms.",
                "Training": "None.",
                "Inference": "Deterministic path rule.",
                "Uncertainty": "None beyond data quality.",
            },
            {
                "Approach": "Linear / time decay / position",
                "Role": "Rule-based comparators",
                "Pros": "Path-aware, easy to audit, stable.",
                "Cons": "Heuristic weights not learned from outcomes.",
                "Training": "None.",
                "Inference": "Deterministic path rule.",
                "Uncertainty": "Config sensitivity only.",
            },
            {
                "Approach": "Markov removal effect",
                "Role": "Data-driven advanced model",
                "Pros": "Uses journey transitions and conversion probability loss.",
                "Cons": "Needs sufficient paths, can be unstable in sparse segments.",
                "Training": "Estimate transition matrix and absorption probabilities from historical journeys.",
                "Inference": "Apply channel importance to conversion paths.",
                "Uncertainty": "Bootstrap CI and confidence label.",
            },
        ]
    )
    st.dataframe(approaches, hide_index=True, use_container_width=True)

    st.markdown(
        """
        **Deterministic fallback:** If Markov confidence is low, consent coverage is poor, conversion probability is too
        small, or bootstrap variance is high, the engine falls back to linear attribution and suppresses actionable
        budget changes. The fallback reason is visible in the dashboard and audit log.
        """
    )

    st.subheader("End-to-end data flow and evidence inspection")
    st.markdown(
        """
        1. Raw synthetic events/conversions are generated with known ground-truth shares.
        2. Validator separates blocking failures from warnings.
        3. Canonical records are stored in SQLite; quarantine retained.
        4. Path builder joins pre-conversion touches within window and consent policy.
        5. Baseline models compute path credits.
        6. Markov trainer estimates channel removal effects from journeys and reports bootstrap CI.
        7. Decision scorer combines attributed revenue, cost, margin, confidence, disagreement, and risk flags.
        8. Dashboard shows observed, estimated, simulated, and predicted values with drill-down evidence.
        9. Reviewer can inspect conversion timeline, model credit by model, data-quality flags, and audit history.
        """
    )

    st.subheader("Scale without premature complexity")
    scale = pd.DataFrame(
        [
            {"Concern": "Batching", "Prototype approach": "Process 9k journeys in memory; batch inserts to SQLite.", "Production evidence to split": "Batch size > millions or SLA requires streaming."},
            {"Concern": "Caching", "Prototype approach": "st.cache_data for generation/validation; session store for model results.", "Production evidence to split": "Repeated expensive model training across tenants."},
            {"Concern": "Queues", "Prototype approach": "Simulated live tick list; no external broker.", "Production evidence to split": "Multi-source ingestion spikes and replay requirements."},
            {"Concern": "Retries/idempotency", "Prototype approach": "Deterministic event IDs and duplicate quarantine.", "Production evidence to split": "Network-facing adapters with partial failures."},
            {"Concern": "Pagination", "Prototype approach": "Dashboard limits tables to top N and search.", "Production evidence to split": "Large result sets and API consumers."},
            {"Concern": "Backpressure", "Prototype approach": "Live history trimmed; simulation speed control.", "Production evidence to split": "Real-time streams exceeding consumer capacity."},
            {"Concern": "Partitioning", "Prototype approach": "Platform/date filters in pandas.", "Production evidence to split": "Tenant or region isolation, petabyte-scale event lake."},
            {"Concern": "Graceful degradation", "Prototype approach": "Fallback to linear/Hold when confidence low.", "Production evidence to split": "Always required."},
        ]
    )
    st.dataframe(scale, hide_index=True, use_container_width=True)

    st.markdown(
        """
        **Initial form:** modular monolith.  
        **Split services when measured evidence appears:** >50M events/day, p95 attribution latency >5s,
        independent team ownership, multi-tenant isolation, or streaming replay requirements.
        """
    )

    st.subheader("Page 2 acceptance checks")
    st.markdown(
        """
        - [ ] Architecture includes source adapters, ingestion, validation, canonical model, features, baselines, model, scoring, API/dashboard, audit, observability.
        - [ ] Ownership and failure boundaries are explicit.
        - [ ] Data contracts include field names, types, units, identifiers, timestamps, provenance, quality flags, versioning, retention.
        - [ ] Missing, duplicate, late, malformed, outlier, schema-change handling is specified.
        - [ ] Transparent baseline is chosen before advanced ML.
        - [ ] At least two approaches are compared with trade-offs.
        - [ ] Training vs inference is defined.
        - [ ] Uncertainty/confidence and deterministic fallback are specified.
        - [ ] Mermaid architecture and sequence diagrams are present.
        - [ ] Reviewer can inspect evidence behind each result.
        - [ ] Scale plan avoids premature complexity and states split evidence.
        """
    )


def render_page3() -> None:
    st.header("🛠️ Page 3 — Technology Stack, Folder Structure, and Build Steps")

    st.subheader("Python-first, CPU-compatible stack")
    stack = pd.DataFrame(
        [
            {"Tool": "Python 3.10+", "Purpose": "Core implementation language.", "Justification": "Windows-friendly, scientific ecosystem, no paid services."},
            {"Tool": "Streamlit", "Purpose": "Dashboard and interactive prototype.", "Justification": "Single-file app, rapid reviewable UI, local CPU."},
            {"Tool": "pandas", "Purpose": "Tabular ingestion, validation, aggregation.", "Justification": "Maintained, open-source, sufficient for prototype scale."},
            {"Tool": "NumPy", "Purpose": "Linear algebra for Markov absorption and bootstrap.", "Justification": "Fast CPU numerical operations."},
            {"Tool": "Plotly", "Purpose": "Interactive charts.", "Justification": "Open-source, works offline after install."},
            {"Tool": "Pillow", "Purpose": "Frame rendering for demo video.", "Justification": "Lightweight image generation without GPU."},
            {"Tool": "OpenCV headless", "Purpose": "Encode moving demo video.", "Justification": "CPU-only video writing; optional fallback to GIF."},
            {"Tool": "SQLite", "Purpose": "Persistent canonical records, quarantine, audit.", "Justification": "Local, no server, appropriate for prototype."},
        ]
    )
    st.dataframe(stack, hide_index=True, use_container_width=True)

    st.markdown(
        """
        **Explicit constraints satisfied**
        - No API keys.
        - No paid APIs.
        - No proprietary cloud services required.
        - Core demo runs on CPU and Intel integrated graphics.
        - Synthetic data only.
        """
    )

    st.subheader("Complete repository layout for productionization")
    folders = pd.DataFrame(
        [
            {"/src": "Application entrypoint and Streamlit composition.", "Boundary": "UI only; no business logic hidden in widgets."},
            {"/api": "FastAPI service adapters for future production.", "Boundary": "Request validation, auth hooks, DTOs."},
            {"/domain": "Canonical entities, enums, value objects.", "Boundary": "No I/O; pure domain models."},
            {"/data": "Dataset loaders, synthetic generators, source adapters.", "Boundary": "External systems isolated here."},
            {"/schemas": "Pydantic/data-contract definitions.", "Boundary": "Versioned contracts and validation."},
            {"/models": "Attribution models: rules, Markov, evaluation helpers.", "Boundary": "Deterministic, testable model code."},
            {"/services": "Path building, scoring, recommendations, audit services.", "Boundary": "Orchestrates domain and models."},
            {"/dashboard": "Streamlit pages, charts, tables, demo controls.", "Boundary": "Presentation and interaction."},
            {"/configs": "Settings, defaults, environment templates.", "Boundary": "No secrets committed."},
            {"/scripts": "PowerShell/Bash setup, data generation, demo export.", "Boundary": "Reproducible operations."},
            {"/migrations": "SQLite/Postgres schema migrations.", "Boundary": "Versioned storage evolution."},
            {"/docs": "PRD, architecture, data dictionary, license inventory.", "Boundary": "Reviewable artifacts."},
            {"/tests/unit": "Parsing, validation, transformations, model units.", "Boundary": "Fast deterministic tests."},
            {"/tests/integration": "Persistence, API validation, retries, migration.", "Boundary": "Local services only."},
            {"/tests/e2e": "Ingestion to user-visible result.", "Boundary": "Full demo workflow."},
        ]
    )
    st.dataframe(folders, hide_index=True, use_container_width=True)
    st.caption("This deliverable intentionally provides a single app.py prototype, while documenting the target modular repository for enterprise handoff.")

    st.subheader("Milestone build steps and exact PowerShell commands")
    st.code(
        r"""
# 1. Create project folder
New-Item -ItemType Directory -Force -Path .\cross-platform-attribution-engine
Set-Location .\cross-platform-attribution-engine

# 2. Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Run the Streamlit prototype
streamlit run app.py

# 5. Optional: create target modular folders for future productionization
$dirs = @(
  "src","api","domain","data","schemas","models","services","dashboard",
  "configs","scripts","migrations","docs","tests/unit","tests/integration","tests/e2e"
)
foreach ($d in $dirs) { New-Item -ItemType Directory -Force -Path $d }

# 6. Optional: initialize git
git init
git add requirements.txt app.py
git commit -m "feat: cross-platform attribution engine prototype"
        """.strip(),
        language="powershell",
    )

    st.subheader("Working README excerpt")
    st.markdown(
        r"""
        # Cross-Platform Attribution Engine

        Run locally on Windows:

        ```powershell
        python -m venv .venv
        .\.venv\Scripts\Activate.ps1
        pip install -r requirements.txt
        streamlit run app.py
        ```

        The app automatically:
        - generates synthetic cross-platform ad events and conversions
        - validates schemas and quarantines bad records
        - builds attribution paths
        - runs baseline and Markov attribution
        - starts a live simulated event stream
        - generates a downloadable moving demo video
        - provides tests, evaluation, audit history, and governance controls

        No API keys are required.
        """
    )

    st.subheader("License inventory")
    licenses = pd.DataFrame(
        [
            {"Package": "streamlit", "License": "Apache-2.0", "Commercial use": "Yes", "Attribution": "Include notice in distributed builds."},
            {"Package": "pandas", "License": "BSD-3-Clause", "Commercial use": "Yes", "Attribution": "Include copyright notice."},
            {"Package": "numpy", "License": "BSD-3-Clause", "Commercial use": "Yes", "Attribution": "Include copyright notice."},
            {"Package": "plotly", "License": "MIT", "Commercial use": "Yes", "Attribution": "Include copyright notice."},
            {"Package": "pillow", "License": "MIT-CMU / HPND", "Commercial use": "Yes", "Attribution": "Include license text."},
            {"Package": "opencv-python-headless", "License": "Apache-2.0", "Commercial use": "Yes", "Attribution": "Include notice."},
            {"Package": "synthetic dataset", "License": "Generated for demo", "Commercial use": "Yes", "Attribution": "No external dataset obligations."},
        ]
    )
    st.dataframe(licenses, hide_index=True, use_container_width=True)
    st.caption("Transitive licenses should be verified at build time with a tool such as pip-licenses before enterprise distribution.")

    st.subheader(".env.example")
    st.code(
        """
# No secrets are required for the synthetic CPU-only demo.
# Placeholders only. Do not commit real credentials.
APP_ENV=local
DB_PATH=./cross_platform_attribution_demo.db
SYNTHETIC_SEED=20261107
RETENTION_DAYS=180
ENABLE_EXTERNAL_WRITES=false
        """.strip(),
        language="ini",
    )

    st.subheader("Data provenance")
    st.markdown(
        """
        - All demo data is synthetically generated in-app using a fixed random seed.
        - No real customer, advertiser, or platform data is used.
        - Ground-truth incremental shares are embedded only for controlled evaluation.
        - Source systems are simulated names such as `google_ads_api`, `meta_marketing_api`, and `internal_ets`.
        """
    )

    st.subheader("Page 3 acceptance checks")
    st.markdown(
        """
        - [ ] Python-first implementation compatible with Windows 10/11.
        - [ ] Core prototype runs on CPU and Intel integrated graphics.
        - [ ] No paid APIs or proprietary cloud services required.
        - [ ] Maintained open-source tools selected with project-specific justification.
        - [ ] Complete repository layout explained with module boundaries.
        - [ ] Milestones and exact PowerShell commands provided.
        - [ ] Working README provided.
        - [ ] Dependency license inventory provided.
        - [ ] .env.example contains placeholders only and no secrets.
        - [ ] Data provenance documented.
        """
    )


def render_page4(store: Dict[str, Any], bundle, settings: Dict[str, Any]) -> None:
    st.header("✅ Page 4 — Testing, Evaluation, Security, and Reliability")

    st.subheader("Automated test suite")
    if st.button("Run required tests"):
        with st.spinner("Running unit, property, integration, and e2e tests..."):
            st.session_state.test_results = run_self_tests()
        log_audit("test_suite_run", {"count": len(st.session_state.test_results)})

    if "test_results" in st.session_state:
        tr = st.session_state.test_results
        passed = int((tr["Status"] == "PASS").sum())
        failed = int((tr["Status"] != "PASS").sum())
        c1, c2 = st.columns(2)
        c1.metric("Passed", passed)
        c2.metric("Failed / errors", failed)
        st.dataframe(tr, hide_index=True, use_container_width=True)
        if failed == 0:
            st.success("All required local tests passed.")
        else:
            st.error("Some tests failed. Review details before acceptance.")
    else:
        st.info("Click 'Run required tests' to execute the local test suite.")

    st.subheader("Controlled evaluation")
    st.markdown(
        """
        The evaluation uses a chronological train/validation/test split and compares model output to the
        synthetic ground-truth incremental shares used to generate the demo data. This is not a claim of
        real-world causal lift; it is a reproducible way to test whether the engine can recover known structure
        and whether uncertainty/fallback behavior is sane.
        """
    )
    if st.button("Run controlled evaluation"):
        with st.spinner("Running chronological evaluation..."):
            st.session_state.eval_result = run_controlled_evaluation(store, bundle, settings)
        log_audit("controlled_evaluation_run", {})

    if "eval_result" in st.session_state:
        er = st.session_state.eval_result
        if er is None:
            st.warning("Not enough test conversions for controlled evaluation under current settings.")
        else:
            table, metrics, fig = er
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Markov MAE", f"{metrics['markov_mae']:.4f}")
            m2.metric("Last-click MAE", f"{metrics['last_click_mae']:.4f}")
            m3.metric("Spearman", f"{metrics['spearman']:.3f}")
            m4.metric("Regret proxy", f"{metrics['regret_proxy']:.4f}")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(table, hide_index=True, use_container_width=True)
            st.json(metrics)
    else:
        st.info("Click 'Run controlled evaluation' to generate a reviewable evaluation artifact.")

    st.subheader("Metrics and formulas")
    st.markdown(
        """
        - **Channel share error:** `MAE = mean(|estimated_share - true_share|)` across platforms.
        - **RMSE:** `sqrt(mean((estimated_share - true_share)^2))`.
        - **Rank quality:** Spearman correlation between estimated and true platform shares.
        - **Regret proxy:** `0.5 * sum(|estimated_share - true_share|)`, an L1 allocation-distance heuristic.
        - **Calibration / uncertainty:** bootstrap 95% CI width for Markov importance; wide CI lowers confidence.
        - **Reliability:** ingestion success rate, quarantine rate, p95 processing latency, audit completeness.
        - **What a metric cannot prove:** aggregate share accuracy does not prove incremental causality without geo/PSA holdouts.
        """
    )

    st.subheader("Security, privacy, and safe failure")
    sec = pd.DataFrame(
        [
            {"control": "Least privilege", "implementation": "Local SQLite file; no cloud credentials; no external writes."},
            {"control": "Input validation", "implementation": "Typed validation, enum checks, timestamp bounds, numeric ranges."},
            {"control": "Secrets management", "implementation": "No secrets required; .env.example contains placeholders only."},
            {"control": "Privacy", "implementation": "Synthetic pseudo IDs, hashed identifiers, consent flags, no real PII."},
            {"control": "Retention", "implementation": "Demo data limited to 90 days; quarantine retained for audit; documented 180-day max."},
            {"control": "Audit logging", "implementation": "Model runs, test runs, approvals, video generation logged to SQLite."},
            {"control": "Human approval", "implementation": "Spend changes, model promotion, data-source onboarding, retention changes require approval."},
            {"control": "Kill switch", "implementation": "Safe mode forces Hold recommendations and suppresses live actions."},
            {"control": "Safe failure", "implementation": "Low confidence or poor data quality triggers deterministic linear fallback and investigation flag."},
        ]
    )
    st.dataframe(sec, hide_index=True, use_container_width=True)

    st.subheader("Page 4 acceptance checks")
    st.markdown(
        """
        - [ ] Unit tests cover parsing, validation, transformations, core calculations, model behavior, boundaries, determinism, and errors.
        - [ ] Property tests verify credit shares sum to one and are non-negative.
        - [ ] Integration tests verify SQLite persistence and API-like validation boundaries locally.
        - [ ] E2E test runs ingestion → validation → paths → attribution → aggregation → recommendations.
        - [ ] Controlled evaluation uses chronological splits and reports uncertainty.
        - [ ] Security controls and human-approval gates are visible and testable.
        """
    )


def render_value_calculator() -> None:
    st.header("💰 Business Value Calculator")
    st.caption("User-entered assumptions only. This does not fabricate customer results or promise commercial success.")

    c1, c2, c3 = st.columns(3)
    monthly_spend = c1.number_input("Monthly ad spend (USD)", min_value=1000.0, value=250000.0, step=5000.0)
    monthly_conversions = c1.number_input("Monthly conversions", min_value=10.0, value=10000.0, step=100.0)
    current_cac = c2.number_input("Current blended CAC (USD)", min_value=1.0, value=25.0, step=1.0)
    improvement_low = c2.number_input("Low improvement (%)", min_value=0.0, value=1.0, step=0.5)
    improvement_central = c3.number_input("Central improvement (%)", min_value=0.0, value=4.0, step=0.5)
    improvement_high = c3.number_input("High improvement (%)", min_value=0.0, value=8.0, step=0.5)

    c4, c5, c6 = st.columns(3)
    adoption = c4.number_input("Adoption rate (%)", min_value=0.0, max_value=100.0, value=60.0, step=5.0)
    one_time_cost = c5.number_input("One-time build/integration cost (USD)", min_value=0.0, value=80000.0, step=5000.0)
    monthly_run_cost = c6.number_input("Monthly run/support cost (USD)", min_value=0.0, value=2500.0, step=250.0)

    if improvement_central > 15.0:
        st.warning("Improvement above 15% is aggressive for attribution-only changes. Validate with geo/PSA holdouts before using in a business case.")

    def scenario(imp_pct: float) -> Dict[str, float]:
        effective = imp_pct / 100.0 * (adoption / 100.0)
        monthly_savings = monthly_spend * effective
        annual_savings = monthly_savings * 12.0
        annual_run = monthly_run_cost * 12.0
        net_annual = annual_savings - annual_run
        monthly_net = monthly_savings - monthly_run_cost
        payback_months = (one_time_cost / monthly_net) if monthly_net > 0 else float("inf")
        new_cac = current_cac * (1.0 - effective) if effective < 1.0 else 0.0
        additional_conversions = (monthly_spend / new_cac - monthly_conversions) if new_cac > 0 else 0.0
        return {
            "scenario": f"{imp_pct:.1f}% gross improvement",
            "effective_improvement_pct": effective * 100.0,
            "monthly_savings_usd": monthly_savings,
            "annual_savings_usd": annual_savings,
            "net_annual_value_usd": net_annual,
            "payback_months": payback_months,
            "modeled_new_cac_usd": new_cac,
            "modeled_additional_conversions_per_month": additional_conversions,
        }

    rows = [scenario(x) for x in [improvement_low, improvement_central, improvement_high, improvement_central * 0.5, min(improvement_high * 1.5, 20.0)]]
    df = pd.DataFrame(rows)
    df["Monthly savings"] = df["monthly_savings_usd"].map(fmt_usd)
    df["Annual net value"] = df["net_annual_value_usd"].map(fmt_usd)
    df["Payback months"] = df["payback_months"].map(lambda x: "Not recoverable" if math.isinf(x) else f"{x:.1f}")
    df["Modeled new CAC"] = df["modeled_new_cac_usd"].map(fmt_usd)

    st.subheader("Sensitivity analysis")
    st.dataframe(
        df[["scenario", "effective_improvement_pct", "Monthly savings", "Annual net value", "Payback months", "Modeled new CAC"]],
        hide_index=True,
        use_container_width=True,
    )

    st.markdown(
        """
        **Guardrails**
        - Value is hypothetical and driven by user assumptions.
        - Attribution improvement is not guaranteed incrementality.
        - Validate with controlled geo experiments or holdout tests before committing capital.
        - Operating costs should include data engineering, governance, incident response, and model monitoring.
        """
    )


def render_page5(store: Dict[str, Any], bundle, settings: Dict[str, Any]) -> None:
    st.header("🚀 Page 5 — Live Demo, Deployment, and Acceptance Criteria")

    st.subheader("Reproducible 5–10 minute demo script")
    st.markdown(
        """
        1. **Setup (30s)**  
           Run `streamlit run app.py`. The app auto-loads synthetic data and starts the live simulation.

        2. **Normal scenario (90s)**  
           Open **Live Demo**. Show cumulative revenue, cost, ROAS, live attributed revenue, and generated video.

        3. **Difficult edge case (90s)**  
           Open **Attribution Workbench**. Switch to a conversion with mixed click/view touches. Show timeline,
           model credit differences, and Markov CI.

        4. **Failure/recovery scenario (90s)**  
           In sidebar, enable **Force deterministic fallback** or toggle **Require consent**.
           Show fallback warning, Hold/Investigate recommendations, and audit log entry.

        5. **Explanation of result (60s)**  
           Show data-quality status, quarantine rate, disagreement vs last-click, and human-approval requirement.

        6. **Acceptance (30s)**  
           Run tests and controlled evaluation in Page 4. Download demo video and audit artifacts.
        """
    )

    st.subheader("Exact setup/run commands and expected outputs")
    st.code(
        r"""
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
        """.strip(),
        language="powershell",
    )
    st.markdown(
        """
        **Expected output**
        - Browser opens Streamlit app.
        - Live Demo tab shows non-blank dashboard within seconds.
        - Synthetic dataset loads with spinner messages.
        - Live simulation advances automatically.
        - A downloadable moving demo file appears (`mp4`, `avi`, or `gif` fallback).
        - Page 4 test suite can run locally without paid services.
        """
    )

    st.subheader("Professional dashboard requirements")
    dash = pd.DataFrame(
        [
            {"Feature": "Overview metrics", "Implemented": "Revenue, cost, conversions, ROAS, CAC, data quality."},
            {"Feature": "Filters", "Implemented": "Lookback window, include view-through, require consent, platform filter, model selector."},
            {"Feature": "Drill-downs", "Implemented": "Conversion timeline, credit by model, channel performance."},
            {"Feature": "Trends", "Implemented": "Daily attributed revenue and live cumulative trend."},
            {"Feature": "Data-quality status", "Implemented": "Consent coverage, quarantine rate, late rate, fraud rate, outlier rate."},
            {"Feature": "Model/algorithm details", "Implemented": "Markov importance, bootstrap CI, fallback reason."},
            {"Feature": "Explanations", "Implemented": "Rationale, risk flags, evidence view."},
            {"Feature": "Configuration", "Implemented": "Sidebar engine controls and safe mode."},
            {"Feature": "Audit history", "Implemented": "SQLite audit log displayed in Audit & Risk tab."},
            {"Feature": "Value labels", "Implemented": "Observed, estimated, simulated, predicted labels used in UI and video."},
        ]
    )
    st.dataframe(dash, hide_index=True, use_container_width=True)

    st.subheader("Local Windows setup and optional Docker Compose")
    st.markdown("**Local Windows**")
    st.code(
        r"""
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
        """.strip(),
        language="powershell",
    )

    st.markdown("**Optional Docker Compose sketch**")
    st.code(
        """
# docker-compose.yml sketch for future productionization
services:
  attribution-app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
    environment:
      - APP_ENV=demo
      - ENABLE_EXTERNAL_WRITES=false
        """.strip(),
        language="yaml",
    )

    st.subheader("Health checks, logs, migrations, backup/restore, rollback")
    ops = pd.DataFrame(
        [
            {"Topic": "Health check", "Prototype": "Dashboard renders, test suite passes, DB file writable.", "Production": "/healthz endpoint, DB connectivity, queue lag."},
            {"Topic": "Structured logs", "Prototype": "Audit table with JSON details.", "Production": "JSON logs with trace IDs and retention."},
            {"Topic": "Migrations", "Prototype": "SQLite tables replaced on demo load.", "Production": "Versioned Alembic/Flyway migrations."},
            {"Topic": "Backup/restore", "Prototype": "Copy SQLite file.", "Production": "Snapshot, PITR, restore drills."},
            {"Topic": "Rollback", "Prototype": "Reset session state and rerun prior seed.", "Production": "Blue/green or canary model/API rollback."},
            {"Topic": "Resource estimates", "Prototype": "~16 GB RAM CPU laptop; 9k journeys in seconds.", "Production": "Profile per event volume and tenant."},
            {"Topic": "Troubleshooting", "Prototype": "Run tests, inspect quarantine, check fallback reason.", "Production": "Dashboards, alerts, on-call runbooks."},
        ]
    )
    st.dataframe(ops, hide_index=True, use_container_width=True)

    st.subheader("Transparent business-value calculator")
    st.markdown("Open the **Value Calculator** tab. It uses user-entered assumptions and shows sensitivity, payback, and guardrails. It does not fabricate customer results.")

    st.subheader("Deliverables checklist")
    deliv = pd.DataFrame(
        [
            {"Deliverable": "Runnable source code", "Status": "app.py and requirements.txt provided."},
            {"Deliverable": "Architecture docs", "Status": "Page 2 tab with diagrams and boundaries."},
            {"Deliverable": "API docs", "Status": "Service boundary and future FastAPI contract described."},
            {"Deliverable": "Data dictionary", "Status": "Page 2 data contracts and example records."},
            {"Deliverable": "License inventory", "Status": "Page 3 license table."},
            {"Deliverable": "Test report", "Status": "Page 4 runnable local test suite."},
            {"Deliverable": "Demo script", "Status": "Page 5 and Live Demo tab."},
            {"Deliverable": "Known limitations", "Status": "Below."},
            {"Deliverable": "Future roadmap", "Status": "Below."},
        ]
    )
    st.dataframe(deliv, hide_index=True, use_container_width=True)

    st.subheader("Known limitations")
    st.markdown(
        """
        - Synthetic data only; not proof of real-world causal lift.
        - Markov model is a simplified removal-effect implementation.
        - Identity resolution is pseudo-ID based and not production-grade.
        - No external ad-platform writes or authentication.
        - Single-file Streamlit prototype is not a multi-tenant service.
        - Video generation may fall back to GIF if OpenCV codec unavailable.
        """
    )

    st.subheader("Future roadmap")
    st.markdown(
        """
        1. Connect read-only ad-platform APIs with OAuth and tenant isolation.
        2. Replace SQLite with Postgres/DuckDB warehouse and event lake.
        3. Add dbt models, data contracts, and automated data-quality tests.
        4. Integrate geo/PSA holdout experiments for causal incrementality.
        5. Expose FastAPI service with authN/authZ, rate limits, and audit webhooks.
        6. Add model registry, drift monitoring, and controlled rollout.
        7. Production dashboard with SLOs, alerting, backup/DR, and change management.
        """
    )

    st.subheader("Final acceptance criteria")
    st.markdown(
        """
        - [ ] Fresh developer can reproduce demo from README.
        - [ ] App starts with non-blank dashboard and auto-loaded realistic demo data.
        - [ ] Simulation starts automatically and allows interaction.
        - [ ] Downloadable moving demo file is generated in-app.
        - [ ] All required local tests pass.
        - [ ] Controlled evaluation artifact is produced.
        - [ ] Data contracts, failure cases, and acceptance checks are reviewable.
        - [ ] Human approval gates and audit history are visible.
        - [ ] No API keys or paid services are required.
        """
    )


def render_audit_risk(store: Dict[str, Any], settings: Dict[str, Any]) -> None:
    st.header("🧾 Audit, Risk, and Human Approval")

    st.subheader("Risk register")
    st.dataframe(risk_register_df(), hide_index=True, use_container_width=True)

    st.subheader("Actions requiring human approval")
    st.markdown(
        """
        - Any non-zero budget increase/decrease recommendation.
        - Model promotion from fallback to data-driven production scoring.
        - Onboarding a new ad-platform data source.
        - Changing retention or privacy policy.
        - Disabling consent filtering or safe mode.
        - Writing to external ad platforms (not implemented in prototype).
        """
    )

    st.subheader("Recommendation approval workflow")
    recs = store["recommendations"]
    actionable = recs[recs["requires_human_approval"] == True]  # noqa: E712
    if actionable.empty:
        st.success("No recommendations currently require approval under current settings.")
    else:
        platform = st.selectbox("Select platform recommendation", actionable["platform"].tolist())
        comment = st.text_input("Reviewer comment", key="approval_comment")
        confirm = st.checkbox("I confirm this is a human-reviewed advisory action, not an automated spend change.", key="approval_confirm")
        c1, c2 = st.columns(2)
        if c1.button("Approve advisory recommendation", disabled=not confirm):
            log_audit(
                "recommendation_approved",
                {"platform": platform, "comment": comment, "settings": settings},
                approval_status="approved_by_human",
            )
            st.success("Approval logged. No external spend change was made.")
        if c2.button("Reject recommendation", disabled=not confirm):
            log_audit(
                "recommendation_rejected",
                {"platform": platform, "comment": comment, "settings": settings},
                approval_status="rejected_by_human",
            )
            st.info("Rejection logged.")

    st.subheader("Audit history")
    try:
        conn = get_db_conn()
        audit_df = pd.read_sql_query("SELECT ts, actor, action, approval_status, details_json FROM audit_log ORDER BY id DESC LIMIT 200", conn)
        st.dataframe(audit_df, hide_index=True, use_container_width=True)
    except Exception as e:
        st.warning(f"Audit history unavailable: {e}")


def render_sidebar_settings() -> Dict[str, Any]:
    st.sidebar.title("⚙️ Engine Controls")
    st.sidebar.caption("Synthetic, CPU-first, governed prototype")

    window_days = st.sidebar.slider("Attribution lookback (days)", 1, 28, 7, key="window_days")
    include_view = st.sidebar.checkbox("Include view-through touches", value=True, key="include_view")
    require_consent = st.sidebar.checkbox("Require consent for model", value=True, key="require_consent")
    half_life_days = st.sidebar.slider("Time-decay half-life (days)", 0.5, 14.0, 2.0, step=0.5, key="half_life_days")
    model_display = st.sidebar.selectbox("Primary attribution model", list(MODEL_MAP.keys()), index=0, key="model_display")
    target_roas = st.sidebar.number_input("Target ROAS", min_value=0.5, max_value=10.0, value=2.0, step=0.1, key="target_roas")
    min_confidence = st.sidebar.selectbox("Minimum confidence for action", ["high", "medium", "low"], index=1, key="min_confidence")
    force_fallback = st.sidebar.checkbox("Force deterministic fallback", value=False, key="force_fallback")
    safe_mode = st.sidebar.checkbox("Safe mode: suppress spend changes", value=False, key="safe_mode")
    auto_live = st.sidebar.checkbox("Auto live simulation", value=True, key="sidebar_auto_live")
    speed = st.sidebar.select_slider("Simulation speed", options=["slow", "normal", "fast"], value="normal", key="sidebar_speed")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **Governance**
        - No external writes.
        - Human approval required.
        - Synthetic data only.
        """
    )

    return {
        "window_days": int(window_days),
        "include_view": bool(include_view),
        "require_consent": bool(require_consent),
        "half_life_days": float(half_life_days),
        "model_display": model_display,
        "model": MODEL_MAP[model_display],
        "target_roas": float(target_roas),
        "min_confidence": min_confidence,
        "force_fallback": bool(force_fallback),
        "safe_mode": bool(safe_mode),
        "auto_live": bool(auto_live),
        "speed": speed,
    }


# ============================================================================
# Main application
# ============================================================================

def main() -> None:
    st.set_page_config(
        page_title="Cross-Platform Attribution Engine",
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🧭 Cross-Platform Attribution Engine")
    st.caption(
        "AI Advertising Intelligence and Revenue Optimization — synthetic, CPU-first, governed prototype. "
        "No API keys. No external ad-platform writes."
    )

    bundle = load_and_validate(SEED)
    settings = render_sidebar_settings()

    if not st.session_state.get("db_persisted"):
        try:
            conn = get_db_conn()
            ds, ev_c, ev_q, co_c, co_q, jo_c, jo_q = bundle
            persist_dataset(conn, ev_c, co_c, jo_c, ev_q, co_q, jo_q)
            log_audit("dataset_loaded", {"seed": SEED, "events": len(ev_c), "conversions": len(co_c)})
            st.session_state.db_persisted = True
        except Exception as e:
            st.warning(f"SQLite persistence warning: {e}")

    store = ensure_attribution_store(bundle, settings)

    tabs = st.tabs(
        [
            "🎬 Live Demo",
            "📄 Page 1 PRD",
            "🧱 Page 2 Architecture",
            "🛠️ Page 3 Stack",
            "✅ Page 4 Tests",
            "🚀 Page 5 Acceptance",
            "📊 Workbench",
            "💰 Value",
            "🧾 Audit & Risk",
        ]
    )

    with tabs[0]:
        render_live_tab(store, bundle, settings)
    with tabs[1]:
        render_page1()
    with tabs[2]:
        render_page2()
    with tabs[3]:
        render_page3()
    with tabs[4]:
        render_page4(store, bundle, settings)
    with tabs[5]:
        render_page5(store, bundle, settings)
    with tabs[6]:
        render_workbench(store, bundle, settings)
    with tabs[7]:
        render_value_calculator()
    with tabs[8]:
        render_audit_risk(store, settings)


if __name__ == "__main__":
    main()