# app_streamlit.py
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load models
with open('random_forest_classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)
with open('random_forest_regressor.pkl', 'rb') as f:
    regressor = pickle.load(f)

st.title("📊 Student Placement & Salary Predictor")
st.write("Enter your details to predict placement chances and expected salary")

# Create input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
        internships = st.number_input("Number of Internships", min_value=0, max_value=10, value=1)
        projects = st.number_input("Projects Count", min_value=0, max_value=20, value=3)
        coding_score = st.slider("Coding Skill Score", 0, 100, 70)
        aptitude_score = st.slider("Aptitude Score", 0, 100, 65)
        comm_score = st.slider("Communication Score", 0, 100, 70)
        
    with col2:
        github = st.number_input("GitHub Repos", min_value=0, max_value=50, value=3)
        linkedin = st.number_input("LinkedIn Connections", min_value=0, max_value=5000, value=300)
        mock_score = st.slider("Mock Interview Score", 0, 100, 65)
        extra_score = st.slider("Extracurricular Score", 0, 100, 60)
        leadership_score = st.slider("Leadership Score", 0, 100, 60)
        study_hours = st.number_input("Study Hours/Day", min_value=0.0, max_value=24.0, value=4.0)
        certifications = st.number_input("Certifications", min_value=0, max_value=20, value=2)
        
    gender = st.selectbox("Gender", ["Male", "Female"])
    branch = st.selectbox("Branch", ["CSE", "IT", "ECE", "EEE", "Civil", "Mechanical"])
    college_tier = st.selectbox("College Tier", ["Tier 1", "Tier 2", "Tier 3"])
    
    submitted = st.form_submit_button("Predict Placement & Salary")

if submitted:
    # Prepare features for model
    input_data = {
        'cgpa': cgpa,
        'internships_count': internships,
        'projects_count': projects,
        'coding_skill_score': coding_score,
        'aptitude_score': aptitude_score,
        'communication_skill_score': comm_score,
        'github_repos': github,
        'linkedin_connections': linkedin,
        'mock_interview_score': mock_score,
        'extracurricular_score': extra_score,
        'leadership_score': leadership_score,
        'study_hours_per_day': study_hours,
        'certifications_count': certifications,
        'gender_Male': 1 if gender == "Male" else 0,
        'branch_Civil': 1 if branch == "Civil" else 0,
        'branch_ECE': 1 if branch == "ECE" else 0,
        'branch_EEE': 1 if branch == "EEE" else 0,
        'branch_IT': 1 if branch == "IT" else 0,
        'branch_Mechanical': 1 if branch == "Mechanical" else 0,
        'college_tier_Tier 2': 1 if college_tier == "Tier 2" else 0,
        'college_tier_Tier 3': 1 if college_tier == "Tier 3" else 0
    }
    
    # Convert to DataFrame
    feature_names = ['cgpa', 'internships_count', 'projects_count', 'coding_skill_score', 
                     'aptitude_score', 'communication_skill_score', 'github_repos', 
                     'linkedin_connections', 'mock_interview_score', 'extracurricular_score',
                     'leadership_score', 'study_hours_per_day', 'certifications_count',
                     'gender_Male', 'branch_Civil', 'branch_ECE', 'branch_EEE', 
                     'branch_IT', 'branch_Mechanical', 'college_tier_Tier 2', 'college_tier_Tier 3']
    
    features = pd.DataFrame([input_data])[feature_names]
    
    # Make predictions
    placement_pred = classifier.predict(features)[0]
    placement_proba = classifier.predict_proba(features)[0][1]
    
    # Display results
    st.markdown("---")
    st.subheader("📈 Prediction Results")
    
    if placement_pred == 1:
        st.success(f"✅ **PLACED** with {placement_proba*100:.1f}% confidence")
        salary_pred = regressor.predict(features)[0]
        st.info(f"💰 **Expected Salary Package:** {salary_pred:.2f} LPA")
        
        # Improvement tips
        if internships < 2:
            st.warning("💡 Tip: Getting 1-2 more internships could increase salary by 30-50%")
        elif certifications < 3:
            st.info("📚 Tip: Adding relevant certifications could boost salary by 10-15%")
    else:
        st.error(f"❌ **NOT PLACED** (Based on your profile)")
        
        # Suggestions
        suggestions = []
        if internships == 0:
            suggestions.append("🔴 **Internships** (Most critical - 71% importance)")
        if certifications < 2:
            suggestions.append("🟡 **Certifications**")
        if cgpa < 7.0:
            suggestions.append("🟡 **Improve CGPA**")
        if coding_score < 60:
            suggestions.append("🟡 **Coding skills**")
        
        if suggestions:
            st.warning("### Recommendations to improve:")
            for s in suggestions:
                st.write(s)
    
    # Feature importance reminder
    with st.expander("ℹ️ How is this calculated?"):
        st.write("""
        **Key factors affecting placement (from our analysis):**
        - Internships: 71% importance
        - Certifications: 7.4% importance  
        - Technical skills: ~2% each
        - CGPA: 1.9% importance
        
        The model achieved 98% accuracy on classification and 62% R² on salary prediction.
        """)