# Crisis_detection.py

def analyze_text(text: str):
    crisis_keywords = [
        "end it all", "kill myself", "suicidal", "want to die", "give up","giving up"
        "can't go on", "worthless", "hopeless", "hurting myself",
        "cut myself", "ending my life", "jump off", "overdose","depress","depressed"
    ]

    text_lower = text.lower()
    detected = any(phrase in text_lower for phrase in crisis_keywords)

    if detected:
        return {
            "result": {
                "crisis_detected": True,
                "severity": "high",
                "recommended_action": (
                    "It sounds like you're going through a really difficult time. "
                    "Please reach out to a mental health professional or call a helpline immediately. "
                    "You're not alone, and there are people who care about you."
                )
            }
        }
    else:
        return {
            "result": {
                "crisis_detected": False,
                "severity": "none",
                "recommended_action": ""
            }
        }

