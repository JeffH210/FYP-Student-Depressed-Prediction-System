def get_feature_advice(feature_name, value):

    if feature_name == "Sleep Duration":
        if value == "Less than 5 hours": 
            return "Your sleep is very short; aim for at least 7-8 hours to improve mental health."
        elif value == "5-6 hours":  
            return "Sleep is slightly low; try to increase sleep duration for better well-being."
        elif value == "7-8 hours":  
            return "Sleep duration is ideal; keep maintaining a regular sleep schedule."
        else:  
            return "Sleep duration is sufficient, ensure it's quality sleep."

    elif feature_name == "Academic Pressure":
        if value >= 4:
            return "High academic pressure detected; consider stress management techniques."
        elif value >= 2:
            return "Moderate academic pressure; try balancing workload with breaks."
        else:
            return "Academic pressure is low; keep maintaining a healthy study routine."

    elif feature_name == "Financial Stress":
        if value >= 4:
            return "Financial stress is high; consider budgeting or seeking financial guidance."
        elif value >= 2:
            return "Moderate financial stress; monitor spending and plan ahead."
        else:
            return "Financial stress is low; maintain your current financial habits."

    elif feature_name == "CGPA":
        if value < 3.0:
            return "Your CGPA is low; focus on effective study strategies rather than stress."
        elif value < 4.0:
            return "CGPA is moderate; balance study and well-being."
        else:
            return "CGPA is high; keep maintaining consistent performance without overworking."

    elif feature_name == "Study Sastisfaction":
        if value <= 2:
            return "Low study sastisfaction; try engaging in subjects you enjoy or seek academic support."
        elif value <= 4:
            return "Moderate sastisfaction; focus on improving study habits for better outcomes."
        else:
            return "High study sastisfaction; maintain your motivation and positive approach."

    elif feature_name == "Suicidal Thoughts":
        if value == "Yes":  
            return "Suicidal thoughts detected; please seek immediate professional help or counseling."
        else:
            return "No suicidal thoughts reported; continue monitoring mental health."

    elif feature_name == "Family History of Mental Illness":
        if value == "Yes":  
            return "Family history of mental illness exists; be proactive in managing stress and mental health."
        else:
            return "No family history reported; maintain healthy habits to prevent stress."

    elif feature_name == "Dietary Habits":
        if value == "Healthy":
            return "Your dietary habits are healthy; continue eating balanced meals."
        elif value == "Moderate":
            return "Diet is moderate; try including more fruits, vegetables, and whole foods."
        elif value == "Unhealthy":
            return "Diet is unhealthy; reduce junk food and maintain balanced meals for mental and physical health."
        
    elif feature_name == "Study Hours":
        if value >= 10:
            return "Excessive study hours detected; ensure you take breaks and avoid burnout."
        elif value >= 6:
            return "Moderate study hours; maintain balance with rest and leisure."
        else:
            return "Study hours are reasonable; maintain a consistent study routine."
    else:
        return "No specific advice available for this feature."
    
