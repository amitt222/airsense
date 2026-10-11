"""
AirSense - Health Advisory & Risk Assessment Module
Author: AirSense Team
Description:
    Maps numerical AQI values to official Central Pollution Control Board (CPCB)
    National Air Quality Index (NAQI) categories and generates tailored, actionable
    medical and lifestyle advisories across four demographic groups:
      1. General Public
      2. Asthma & Respiratory Patients
      3. Children & Elderly
      4. Outdoor Workers
    Also provides an outdoor safety score (0-100) and mask/activity recommendations.
"""

from typing import Any, Dict, List, Optional


# Official CPCB NAQI Categories, Scale Bands, and Standard Hex Color Scheme
CPCB_SCALE: Dict[str, Dict[str, Any]] = {
    "Good": {
        "min": 0,
        "max": 50,
        "color": "#00B050",
        "badge": "success",
        "description": "Minimal impact on health.",
    },
    "Satisfactory": {
        "min": 51,
        "max": 100,
        "color": "#92D050",
        "badge": "success",
        "description": "Minor breathing discomfort to sensitive people.",
    },
    "Moderate": {
        "min": 101,
        "max": 200,
        "color": "#FFC000",
        "badge": "warning",
        "description": "Breathing discomfort to people with lung, asthma, and heart diseases.",
    },
    "Poor": {
        "min": 201,
        "max": 300,
        "color": "#FF6600",
        "badge": "warning",
        "description": "Breathing discomfort to most people on prolonged outdoor exposure.",
    },
    "Very Poor": {
        "min": 301,
        "max": 400,
        "color": "#FF0000",
        "badge": "danger",
        "description": "Respiratory illness on prolonged exposure.",
    },
    "Severe": {
        "min": 401,
        "max": 500,
        "color": "#990033",
        "badge": "danger",
        "description": "Affects healthy individuals and seriously impacts those with existing conditions.",
    },
}

# Targeted Health Guidance by Demographic Group and CPCB Category
HEALTH_ADVISORIES: Dict[str, Dict[str, Dict[str, str]]] = {
    "Good": {
        "general": {
            "summary": "Air quality is ideal for all normal outdoor activities.",
            "activity": "Normal outdoor exercise, jogging, and sports are encouraged.",
            "mask": "No mask required.",
            "indoor": "Keep windows open for natural ventilation.",
        },
        "asthma": {
            "summary": "Air quality poses virtually zero respiratory threat.",
            "activity": "Unrestricted outdoor activities. Maintain routine medications.",
            "mask": "No mask needed.",
            "indoor": "Fresh outdoor air ventilation is beneficial.",
        },
        "elderly": {
            "summary": "Great air quality for morning walks and outdoor leisure.",
            "activity": "Ideal conditions for walking and park activities.",
            "mask": "No mask needed.",
            "indoor": "Enjoy fresh air and natural room ventilation.",
        },
        "outdoor_workers": {
            "summary": "Safe atmospheric conditions for strenuous labor.",
            "activity": "Standard work shifts without respiratory restrictions.",
            "mask": "Standard occupational dust mask only if working in direct construction.",
            "indoor": "N/A",
        },
    },
    "Satisfactory": {
        "general": {
            "summary": "Air quality is acceptable. Pollutants are within manageable limits.",
            "activity": "Standard outdoor activities are completely safe.",
            "mask": "No mask required for general public.",
            "indoor": "Normal room ventilation is fine.",
        },
        "asthma": {
            "summary": "Extremely sensitive individuals may experience minor cough or irritation.",
            "activity": "Outdoor workouts permitted; take breaks if chest tightness occurs.",
            "mask": "Keep a surgical mask handy if in heavy traffic zones.",
            "indoor": "Ensure regular medication adherence.",
        },
        "elderly": {
            "summary": "Minor irritation possible for seniors with chronic bronchitis.",
            "activity": "Morning walks permitted. Stay well hydrated.",
            "mask": "Optional light mask along busy roadways.",
            "indoor": "Normal ventilation.",
        },
        "outdoor_workers": {
            "summary": "Safe for regular outdoor work shifts.",
            "activity": "Normal working conditions.",
            "mask": "Standard dust protection if around construction sites.",
            "indoor": "N/A",
        },
    },
    "Moderate": {
        "general": {
            "summary": "Noticeable haze. Breathing discomfort possible on prolonged exposure.",
            "activity": "Reduce prolonged heavy exertion along congested corridors.",
            "mask": "Surgical or cloth mask recommended during peak commute hours.",
            "indoor": "Close windows during peak evening traffic.",
        },
        "asthma": {
            "summary": "Increased likelihood of bronchospasms, coughing, and wheezing.",
            "activity": "Avoid intense outdoor cardio; transition workouts indoors.",
            "mask": "Wear a well-fitted N95 mask outdoors.",
            "indoor": "Keep rescue inhaler readily accessible at all times.",
        },
        "elderly": {
            "summary": "Elevated particulate matter can stress cardiovascular systems.",
            "activity": "Limit morning walks; reschedule to sunny afternoons when haze lifts.",
            "mask": "Wear a comfortable filtering mask when traveling outdoors.",
            "indoor": "Keep windows closed in the early morning and late evening.",
        },
        "outdoor_workers": {
            "summary": "Prolonged shifts increase cumulative particulate inhalation.",
            "activity": "Take periodic rest breaks in shaded or sheltered areas.",
            "mask": "Wear an N95 or dual-filter dust mask throughout shift.",
            "indoor": "Stay hydrated to assist mucosal clearing.",
        },
    },
    "Poor": {
        "general": {
            "summary": "Unhealthy air. Prolonged exposure causes cough and throat irritation.",
            "activity": "Curtail prolonged outdoor running and strenuous sports.",
            "mask": "N95 / FFP2 mask strongly advised when outside.",
            "indoor": "Keep windows closed; run HEPA air purifiers if available.",
        },
        "asthma": {
            "summary": "High risk of acute asthma exacerbation and respiratory distress.",
            "activity": "Strictly avoid outdoor exercise. Remain indoors.",
            "mask": "Mandatory N95 mask if stepping outside is unavoidable.",
            "indoor": "Use air purifiers; monitor peak flow readings closely.",
        },
        "elderly": {
            "summary": "Significantly elevated risk of bronchitis and cardiac strain.",
            "activity": "Avoid outdoor walks completely until AQI improves.",
            "mask": "Mandatory N95 mask for any brief doctor/essential trip.",
            "indoor": "Keep doors and windows sealed. Maintain indoor humidity.",
        },
        "outdoor_workers": {
            "summary": "Hazardous occupational exposure over full working day.",
            "activity": "Rotate shifts; reduce heavy manual lifting outdoors.",
            "mask": "Mandatory N95 / KN95 respirator with airtight seal.",
            "indoor": "Provide clean-air rest stations with air filtration.",
        },
    },
    "Very Poor": {
        "general": {
            "summary": "Very unhealthy air. Significant respiratory illness risk on exposure.",
            "activity": "Avoid all non-essential outdoor activity. Exercise indoors only.",
            "mask": "Strict N95 / N99 mask requirement for all outdoor movement.",
            "indoor": "Seal windows and doors; run air purifiers on high speed.",
        },
        "asthma": {
            "summary": "Critical trigger hazard. High probability of emergency care need.",
            "activity": "Absolute prohibition on outdoor activities. Stay indoors.",
            "mask": "N99 / N95 respirator if emergency outdoor transit is needed.",
            "indoor": "Ensure full medication supply; maintain an emergency medical plan.",
        },
        "elderly": {
            "summary": "Severe cardiac and pulmonary hazard for seniors and children.",
            "activity": "Do not leave the home. Keep children from outdoor playgrounds.",
            "mask": "N95 mask mandatory if visiting a healthcare facility.",
            "indoor": "Maintain strict indoor air filtration; wet mop floors to trap dust.",
        },
        "outdoor_workers": {
            "summary": "Dangerous occupational hazard. Severe acute respiratory loading.",
            "activity": "Mandatory reduction of outdoor shifts; postpone heavy construction.",
            "mask": "Industrial-grade N95 / P100 respirator provided by employer.",
            "indoor": "Mandatory indoor air-filtered rest intervals every 2 hours.",
        },
    },
    "Severe": {
        "general": {
            "summary": "Emergency health hazard. Healthy people suffer respiratory injury.",
            "activity": "Emergency alert: Strictly zero outdoor sports, jogging, or walks.",
            "mask": "High-efficiency N99 / P100 certified respirator for any transit.",
            "indoor": "Keep all windows shut; use continuous air purifiers and wet mopping.",
        },
        "asthma": {
            "summary": "Life-threatening pollution spike. Extreme trigger risk.",
            "activity": "Remain completely inside a sealed, air-purified room.",
            "mask": "Airtight N99 respirator mandatory for any medical transit.",
            "indoor": "Continuous medical monitoring; have emergency contact ready.",
        },
        "elderly": {
            "summary": "Dangerous levels triggering acute coronary and pulmonary episodes.",
            "activity": "Stay strictly indoors. No outdoor movement for elderly or toddlers.",
            "mask": "N95 / N99 mask mandatory if evacuation or hospital visit needed.",
            "indoor": "Keep living quarters sealed; use indoor plants and air cleaners.",
        },
        "outdoor_workers": {
            "summary": "Extreme health hazard. Outdoor physical labor should be suspended.",
            "activity": "Halt non-emergency outdoor manual labor per disaster protocols.",
            "mask": "Fitted N99 / P100 respirator if emergency work is required.",
            "indoor": "Relocate workforce to indoor sheltered facilities immediately.",
        },
    },
}


def get_cpcb_category(aqi: float) -> Dict[str, Any]:
    """
    Identifies the official CPCB category, scale parameters, and display properties for an AQI value.

    Args:
        aqi (float): Air Quality Index value.

    Returns:
        Dict[str, Any]: Category name, color, badge, and health impact overview.
    """
    if aqi is None or aqi < 0:
        return {
            "category": "Unknown",
            "color": "#888888",
            "badge": "secondary",
            "description": "No valid AQI reading available.",
            "min": 0,
            "max": 0,
        }

    val = float(aqi)
    if val <= 50:
        cat = "Good"
    elif val <= 100:
        cat = "Satisfactory"
    elif val <= 200:
        cat = "Moderate"
    elif val <= 300:
        cat = "Poor"
    elif val <= 400:
        cat = "Very Poor"
    else:
        cat = "Severe"

    meta = CPCB_SCALE[cat]
    return {
        "category": cat,
        "color": meta["color"],
        "badge": meta["badge"],
        "description": meta["description"],
        "min": meta["min"],
        "max": meta["max"],
    }


def get_health_advisory(
    aqi: float,
    user_group: Optional[str] = None
) -> Dict[str, Any]:
    """
    Returns tailored health recommendations for a specific demographic group or all groups.

    Args:
        aqi (float): Forecasted or observed AQI.
        user_group (Optional[str]): One of 'general', 'asthma', 'elderly', 'outdoor_workers'.
                                   If None, returns advice for all four groups.

    Returns:
        Dict[str, Any]: Formatted advisory dictionary.
    """
    cat_info = get_cpcb_category(aqi)
    cat_name = cat_info["category"]

    if cat_name == "Unknown":
        return {
            "aqi": aqi,
            "category": "Unknown",
            "advisories": {},
        }

    category_advisories = HEALTH_ADVISORIES.get(cat_name, HEALTH_ADVISORIES["Severe"])

    if user_group:
        group_key = user_group.lower().strip()
        # Handle aliases
        alias_map = {
            "general public": "general",
            "asthma patients": "asthma",
            "respiratory": "asthma",
            "children": "elderly",
            "seniors": "elderly",
            "workers": "outdoor_workers",
        }
        group_key = alias_map.get(group_key, group_key)
        group_advice = category_advisories.get(group_key, category_advisories["general"])
        return {
            "aqi": aqi,
            "category": cat_name,
            "color": cat_info["color"],
            "badge": cat_info["badge"],
            "description": cat_info["description"],
            "group": group_key,
            "advice": group_advice,
        }

    return {
        "aqi": aqi,
        "category": cat_name,
        "color": cat_info["color"],
        "badge": cat_info["badge"],
        "description": cat_info["description"],
        "advisories": category_advisories,
    }


def calculate_outdoor_safety_score(
    aqi: float,
    user_group: str = "general"
) -> Dict[str, Any]:
    """
    Computes an intuitive 0-100 'Should I Go Out?' Safety Score and provides a clear verdict.

    Formula:
      - Good (0-50): 95 - 100 (Safe)
      - Satisfactory (51-100): 80 - 94 (Safe with minor caveats)
      - Moderate (101-200): 55 - 79 (Caution)
      - Poor (201-300): 30 - 54 (Unhealthy / Minimize)
      - Very Poor (301-400): 10 - 29 (Hazardous / Avoid)
      - Severe (401-500+): 0 - 9 (Danger / Stay Indoors)

    Adjusted by user group vulnerability factor.

    Args:
        aqi (float): AQI index value.
        user_group (str): Demographic group.

    Returns:
        Dict[str, Any]: Numerical safety score (0-100), verdict, and recommendation.
    """
    val = float(max(0, aqi))

    # Base safety score calculation
    if val <= 50:
        base_score = 100.0 - (val / 50.0) * 10.0  # 100 -> 90
    elif val <= 100:
        base_score = 90.0 - ((val - 50.0) / 50.0) * 15.0  # 90 -> 75
    elif val <= 200:
        base_score = 75.0 - ((val - 100.0) / 100.0) * 25.0  # 75 -> 50
    elif val <= 300:
        base_score = 50.0 - ((val - 200.0) / 100.0) * 25.0  # 50 -> 25
    elif val <= 400:
        base_score = 25.0 - ((val - 300.0) / 100.0) * 18.0  # 25 -> 7
    else:
        base_score = max(0.0, 7.0 - ((val - 400.0) / 100.0) * 7.0)

    # Vulnerability penalty multiplier
    group_penalty = {
        "general": 1.0,
        "outdoor_workers": 1.15,
        "elderly": 1.25,
        "asthma": 1.35,
    }
    penalty = group_penalty.get(user_group.lower(), 1.0)
    adjusted_score = round(max(0.0, min(100.0, 100.0 - ((100.0 - base_score) * penalty))), 1)

    # Verdict derivation
    if adjusted_score >= 80.0:
        verdict = "Safe to Go Out"
        color = "#00B050"
        recommendation = "Great time for outdoor exercise, errands, and leisure."
    elif adjusted_score >= 60.0:
        verdict = "Generally Safe"
        color = "#92D050"
        recommendation = "Safe for general activities; sensitive groups should avoid peak congestion."
    elif adjusted_score >= 40.0:
        verdict = "Caution Advised"
        color = "#FFC000"
        recommendation = "Limit prolonged strenuous outdoor exertion. Wear a mask in traffic."
    elif adjusted_score >= 20.0:
        verdict = "Avoid Non-Essential Outings"
        color = "#FF6600"
        recommendation = "Unhealthy conditions. Exercise indoors and wear an N95 mask outside."
    else:
        verdict = "Hazardous - Stay Indoors"
        color = "#FF0000"
        recommendation = "Dangerous air quality. Keep windows shut and run air purifiers."

    return {
        "safety_score": adjusted_score,
        "verdict": verdict,
        "color": color,
        "recommendation": recommendation,
        "user_group": user_group,
    }


if __name__ == "__main__":
    test_aqis = [35.0, 85.0, 160.0, 260.0, 360.0, 460.0]

    print("================================================================================")
    print("                    AIRSENSE: HEALTH ADVISORY MODULE TEST")
    print("================================================================================")

    for test_aqi in test_aqis:
        cat = get_cpcb_category(test_aqi)
        score_gen = calculate_outdoor_safety_score(test_aqi, "general")
        score_asthma = calculate_outdoor_safety_score(test_aqi, "asthma")
        advice = get_health_advisory(test_aqi, "asthma")

        print(f"\n[AQI = {test_aqi:3.0f}] -> Category: {cat['category']:12s} (Color: {cat['color']})")
        print(f"  Safety Score (General) : {score_gen['safety_score']:4.1f}/100 -> {score_gen['verdict']}")
        print(f"  Safety Score (Asthma)  : {score_asthma['safety_score']:4.1f}/100 -> {score_asthma['verdict']}")
        print(f"  Asthma Advice          : {advice['advice']['summary']}")
        print(f"  Mask Recommendation    : {advice['advice']['mask']}")
