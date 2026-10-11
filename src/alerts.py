"""
AirSense - Automated Alert Notification System
Author: AirSense Team
Description:
    Dispatches automated early-warning health alerts via Email (SMTP) and Telegram
    when forecasted AQI crosses critical regulatory thresholds (e.g., CPCB Poor / Severe).
    Reads credentials securely from environment variables (.env).
    Provides robust fallbacks, rich HTML email templates, and dry-run simulation modes
    for presentation and testing without requiring live credentials.
"""

import os
import sys
from pathlib import Path
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, Optional
from dotenv import load_dotenv
import requests

# Reconfigure stdout to UTF-8 on Windows consoles to cleanly handle emojis
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.advisory import get_cpcb_category, get_health_advisory

# Load environment configuration from .env file at project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# Default Settings from Environment
DEFAULT_ALERT_THRESHOLD = float(os.getenv("AQI_ALERT_THRESHOLD", "200.0"))
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
ALERT_EMAIL_SENDER = os.getenv("ALERT_EMAIL_SENDER", "")
ALERT_EMAIL_PASSWORD = os.getenv("ALERT_EMAIL_PASSWORD", "")
ALERT_EMAIL_RECIPIENT = os.getenv("ALERT_EMAIL_RECIPIENT", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


def is_email_configured() -> bool:
    """Checks whether valid email credentials are provided in the environment."""
    return bool(ALERT_EMAIL_SENDER and ALERT_EMAIL_PASSWORD and ALERT_EMAIL_RECIPIENT)


def is_telegram_configured() -> bool:
    """Checks whether valid Telegram credentials are provided in the environment."""
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def build_alert_message_text(
    city: str,
    forecast_aqi: float,
    category: str,
    horizon_hours: int,
    threshold: float,
    advisories: Dict[str, Any]
) -> str:
    """
    Constructs an urgent plain-text alert suitable for Telegram and SMS.

    Args:
        city (str): City name.
        forecast_aqi (float): Predicted AQI.
        category (str): CPCB category.
        horizon_hours (int): Prediction horizon (24, 48, 72h).
        threshold (float): Triggered threshold.
        advisories (Dict[str, Any]): Health advisories dict.

    Returns:
        str: Formatted message string.
    """
    adv_dict = advisories.get("advisories", {})
    gen_adv = adv_dict.get("general", {}).get("summary", "Take necessary respiratory precautions.")
    asthma_adv = adv_dict.get("asthma", {}).get("summary", "Keep rescue inhalers accessible.")
    mask_adv = adv_dict.get("general", {}).get("mask", "N95 Respirator recommended.")

    return (
        f"🚨 AIRSENSE AIR QUALITY ALERT: {city.upper()} 🚨\n\n"
        f"⚠️ Hazard Level: {category.upper()} (Forecast AQI: {forecast_aqi:.0f})\n"
        f"⏱️ Forecast Horizon: Next {horizon_hours} Hours\n"
        f"🎯 Alert Threshold Exceeded: {threshold:.0f} AQI\n\n"
        f"📌 General Public Advisory:\n"
        f"• {gen_adv}\n"
        f"• Mask Recommendation: {mask_adv}\n\n"
        f"🫁 Sensitive & Asthma Patients:\n"
        f"• {asthma_adv}\n\n"
        f"🔗 Stay safe and check full forecasts on the AirSense Dashboard.\n"
        f"— Automated dispatch by AirSense Monitoring System"
    )


def build_alert_message_html(
    city: str,
    forecast_aqi: float,
    category: str,
    color: str,
    horizon_hours: int,
    threshold: float,
    advisories: Dict[str, Any]
) -> str:
    """
    Constructs an attractive, responsive HTML email alert template.

    Args:
        city (str): City name.
        forecast_aqi (float): Predicted AQI.
        category (str): CPCB category.
        color (str): Hex color code.
        horizon_hours (int): Horizon hours.
        threshold (float): Threshold AQI.
        advisories (Dict[str, Any]): Advisories dictionary.

    Returns:
        str: HTML markup string.
    """
    adv = advisories.get("advisories", {})
    gen_advice = adv.get("general", {})
    asthma_advice = adv.get("asthma", {})
    elderly_advice = adv.get("elderly", {})

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f4f7; margin: 0; padding: 20px; }}
            .container {{ max-width: 600px; margin: 0 auto; background: #ffffff; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
            .header {{ background-color: {color}; color: #ffffff; padding: 24px; text-align: center; }}
            .content {{ padding: 24px; color: #333333; }}
            .badge {{ display: inline-block; background: rgba(0,0,0,0.25); padding: 4px 12px; border-radius: 20px; font-weight: bold; font-size: 14px; }}
            .card {{ background: #fdfdfd; border-left: 4px solid {color}; padding: 12px 16px; margin: 12px 0; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
            .footer {{ background: #eeeeee; padding: 16px; text-align: center; font-size: 12px; color: #777777; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1 style="margin: 0; font-size: 24px;">🚨 AirSense Early Warning Alert</h1>
                <p style="margin: 6px 0 0 0; font-size: 15px;">Critical Air Pollution Detected in <b>{city}</b></p>
                <div style="margin-top: 12px;">
                    <span class="badge">{category.upper()} • {forecast_aqi:.0f} AQI</span>
                </div>
            </div>
            <div class="content">
                <p>The <b>AirSense ML Forecasting Engine</b> predicts that atmospheric air quality in <b>{city}</b> will cross your safe threshold of <b>{threshold:.0f} AQI</b> over the next <b>{horizon_hours} hours</b>.</p>

                <h3 style="color: #222; margin-top: 20px;">🛡️ Targeted Health Actions</h3>

                <div class="card">
                    <b>🏃 General Public:</b>
                    <p style="margin: 4px 0 0 0;">{gen_advice.get('summary', 'Limit outdoor exposure.')}</p>
                    <p style="margin: 4px 0 0 0; font-size: 13px; color: #555;"><b>Mask Requirement:</b> {gen_advice.get('mask', 'N95 Respirator')}</p>
                </div>

                <div class="card">
                    <b>🫁 Asthma & Respiratory Patients:</b>
                    <p style="margin: 4px 0 0 0;">{asthma_advice.get('summary', 'High risk of bronchospasm.')}</p>
                    <p style="margin: 4px 0 0 0; font-size: 13px; color: #555;"><b>Precautions:</b> {asthma_advice.get('indoor', 'Keep inhaler accessible.')}</p>
                </div>

                <div class="card">
                    <b>👴 Elderly & Children:</b>
                    <p style="margin: 4px 0 0 0;">{elderly_advice.get('summary', 'High risk of acute pulmonary strain.')}</p>
                    <p style="margin: 4px 0 0 0; font-size: 13px; color: #555;"><b>Activity:</b> {elderly_advice.get('activity', 'Remain indoors.')}</p>
                </div>
            </div>
            <div class="footer">
                AirSense Platform • College CSE Project • UN SDG 3 & 11 Compliant<br>
                Automated telemetry alert sent from AirSense Intelligence
            </div>
        </div>
    </body>
    </html>
    """


def send_telegram_alert(
    message_text: str,
    bot_token: Optional[str] = None,
    chat_id: Optional[str] = None
) -> bool:
    """
    Dispatches a Telegram message via the official Bot API.

    Args:
        message_text (str): Content to transmit.
        bot_token (Optional[str]): Bot token override.
        chat_id (Optional[str]): Chat ID override.

    Returns:
        bool: True if transmitted successfully, False otherwise.
    """
    token = bot_token or TELEGRAM_BOT_TOKEN
    chat = chat_id or TELEGRAM_CHAT_ID

    if not token or not chat:
        print("[AirSense Alert Warning] Telegram bot token or chat ID not set in .env. Skipping dispatch.")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat,
        "text": message_text,
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        print(f"[AirSense Alert] Telegram alert dispatched successfully to chat ID: {chat}")
        return True
    except Exception as exc:
        print(f"[AirSense Alert Error] Failed sending Telegram alert: {exc}")
        return False


def send_email_alert(
    subject: str,
    text_body: str,
    html_body: Optional[str] = None,
    sender: Optional[str] = None,
    password: Optional[str] = None,
    recipient: Optional[str] = None,
    server_host: Optional[str] = None,
    server_port: Optional[int] = None,
) -> bool:
    """
    Sends an alert email via SMTP with TLS encryption.

    Args:
        subject (str): Email subject.
        text_body (str): Plain text version.
        html_body (Optional[str]): HTML formatted version.

    Returns:
        bool: True if email sent, False otherwise.
    """
    s_email = sender or ALERT_EMAIL_SENDER
    s_pass = password or ALERT_EMAIL_PASSWORD
    r_email = recipient or ALERT_EMAIL_RECIPIENT
    host = server_host or SMTP_SERVER
    port = server_port or SMTP_PORT

    if not s_email or not s_pass or not r_email:
        print("[AirSense Alert Warning] SMTP email credentials not configured in .env. Skipping dispatch.")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"AirSense Alert <{s_email}>"
    msg["To"] = r_email

    msg.attach(MIMEText(text_body, "plain"))
    if html_body:
        msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(host, port, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(s_email, s_pass)
            server.send_message(msg)

        print(f"[AirSense Alert] Email notification delivered to {r_email}")
        return True
    except Exception as exc:
        print(f"[AirSense Alert Error] Failed delivering email alert: {exc}")
        return False


def evaluate_and_dispatch_alerts(
    city: str,
    forecast_aqi: float,
    horizon_hours: int = 24,
    threshold: Optional[float] = None,
    force_send: bool = False
) -> Dict[str, Any]:
    """
    Evaluates whether a forecast AQI crosses the user-specified threshold
    and triggers notifications across all active channels.

    Args:
        city (str): Name of city.
        forecast_aqi (float): Forecasted AQI value.
        horizon_hours (int): 24, 48, or 72 hours.
        threshold (Optional[float]): Threshold value. Defaults to AQI_ALERT_THRESHOLD.
        force_send (bool): If True, bypasses threshold check for testing.

    Returns:
        Dict[str, Any]: Summary execution report.
    """
    limit = threshold if threshold is not None else DEFAULT_ALERT_THRESHOLD
    cat_info = get_cpcb_category(forecast_aqi)
    advisories = get_health_advisory(forecast_aqi)

    crossed = forecast_aqi >= limit or force_send

    result: Dict[str, Any] = {
        "city": city,
        "forecast_aqi": round(forecast_aqi, 1),
        "category": cat_info["category"],
        "threshold": limit,
        "threshold_crossed": crossed,
        "telegram_sent": False,
        "email_sent": False,
        "status": "No alert needed" if not crossed else "Threshold crossed",
    }

    if not crossed:
        return result

    # Build messages
    text_msg = build_alert_message_text(
        city=city,
        forecast_aqi=forecast_aqi,
        category=cat_info["category"],
        horizon_hours=horizon_hours,
        threshold=limit,
        advisories=advisories,
    )
    html_msg = build_alert_message_html(
        city=city,
        forecast_aqi=forecast_aqi,
        category=cat_info["category"],
        color=cat_info["color"],
        horizon_hours=horizon_hours,
        threshold=limit,
        advisories=advisories,
    )

    subject = f"⚠️ AirSense Alert: High AQI ({forecast_aqi:.0f} - {cat_info['category']}) Forecasted in {city}"

    # Dispatch to Telegram
    if is_telegram_configured() or force_send:
        result["telegram_sent"] = send_telegram_alert(text_msg)

    # Dispatch to Email
    if is_email_configured() or force_send:
        result["email_sent"] = send_email_alert(subject, text_msg, html_msg)

    result["message_preview"] = text_msg
    return result


def simulate_alert_dry_run(city: str = "Delhi", test_aqi: float = 278.0) -> None:
    """
    Runs a safe demonstration dry-run printing formatted alert messages and credentials status.
    """
    print("\n" + "=" * 80)
    print("                 AIRSENSE: ALERT MODULE DRY-RUN SIMULATION")
    print("=" * 80)

    print("\n[Credential Configuration Status]")
    print(f"  • Email SMTP Configured    : {'YES (Ready)' if is_email_configured() else 'NO (Set in .env)'}")
    print(f"  • Telegram Bot Configured  : {'YES (Ready)' if is_telegram_configured() else 'NO (Set in .env)'}")
    print(f"  • Default Threshold        : {DEFAULT_ALERT_THRESHOLD} AQI")

    print(f"\n[Simulating Alert Trigger for {city} (Forecast AQI = {test_aqi})]...")
    outcome = evaluate_and_dispatch_alerts(city=city, forecast_aqi=test_aqi, horizon_hours=24, force_send=True)

    print("\n--- Formatted Dispatch Message Preview ---")
    print(outcome.get("message_preview", "N/A"))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    simulate_alert_dry_run()
