import streamlit as st
import pandas as pd

def main():
    st.set_page_config(
        page_title="Leprosy Detection & Risk Assessment",
        page_icon="🩺",
        layout="wide"
    )
    
    st.title("🩺 Leprosy Detection & Risk Assessment")
    st.markdown("**Simple health check - Enter your basic details below**")
    
    # Patient information form
    st.subheader("Patient Information")
    
    with st.form("patient_form"):
        st.markdown("**Basic Information**")
        col1, col2 = st.columns(2)
        
        with col1:
            patient_age = st.number_input("Patient Age", min_value=1, max_value=120, value=30)
            patient_gender = st.selectbox("Patient Gender", ["Male", "Female"])
            
        with col2:
            diabeties = st.selectbox("Diabeties", ["No", "Yes"])
            grade = st.selectbox("Grade", ["0", "1", "2", "3"])
        
        st.markdown("**Foot Measurements (in cm)**")
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("*Right Foot:*")
            length_foot_right = st.number_input("Length of Foot Right", min_value=10.0, max_value=35.0, value=25.0, step=0.1)
            ball_girth_right = st.number_input("Ball Girth Right", min_value=15.0, max_value=35.0, value=22.0, step=0.1)
            in_step_right = st.number_input("In Step Right", min_value=20.0, max_value=35.0, value=26.0, step=0.1)
            
        with col4:
            st.markdown("*Left Foot:*")
            length_foot_left = st.number_input("Length of Foot Left", min_value=10.0, max_value=35.0, value=25.0, step=0.1)
            ball_girth_left = st.number_input("Ball Girth Left", min_value=15.0, max_value=35.0, value=22.0, step=0.1)
            in_step_left = st.number_input("In Step Left", min_value=20.0, max_value=35.0, value=26.0, step=0.1)
        
        submit = st.form_submit_button("Check My Risk", type="primary", use_container_width=True)
        
        if submit:
            # Risk calculation based on patient data and foot measurements
            risk_score = 0
            measurement_alerts = []
            
            # Basic patient information risk factors
            if patient_age > 60 or patient_age < 15:
                risk_score += 15
            if diabeties == "Yes":
                risk_score += 25
            if grade == "1":
                risk_score += 30
            elif grade == "2":
                risk_score += 60
            elif grade == "3":
                risk_score += 90
            
            # Foot measurement analysis for leprosy indicators
            # Check for asymmetry between left and right foot (common in leprosy)
            length_diff = abs(length_foot_right - length_foot_left)
            ball_girth_diff = abs(ball_girth_right - ball_girth_left)
            in_step_diff = abs(in_step_right - in_step_left)
            
            if length_diff > 1.0:  # Significant difference in foot length
                risk_score += 20
                measurement_alerts.append(f"Foot length asymmetry detected: {length_diff:.1f}cm difference")
            
            if ball_girth_diff > 1.0:  # Significant difference in ball girth
                risk_score += 15
                measurement_alerts.append(f"Ball girth asymmetry detected: {ball_girth_diff:.1f}cm difference")
            
            if in_step_diff > 2.0:  # Significant difference in in-step measurement
                risk_score += 10
                measurement_alerts.append(f"In-step asymmetry detected: {in_step_diff:.1f}cm difference")
            
            # Check for unusually small measurements (possible muscle atrophy)
            if length_foot_right < 20 or length_foot_left < 20:
                risk_score += 15
                measurement_alerts.append("Unusually small foot length detected")
            
            if ball_girth_right < 18 or ball_girth_left < 18:
                risk_score += 10
                measurement_alerts.append("Reduced ball girth measurements")
            
            # Show results
            st.markdown("---")
            st.subheader("Risk Assessment Results")
            
            if risk_score >= 80:
                st.error("**HIGH RISK** - Please visit a doctor immediately")
                recommendation = "Visit a healthcare center today"
            elif risk_score >= 40:
                st.warning("**MEDIUM RISK** - Schedule a medical check-up")
                recommendation = "Make an appointment within 1-2 weeks"
            else:
                st.success("**LOW RISK** - Continue routine health care")
                recommendation = "Keep up with regular check-ups"
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Risk Score", f"{risk_score}/100")
            with col_b:
                st.metric("Recommendation", recommendation)
            
            # Show measurement analysis if any alerts
            if measurement_alerts:
                st.subheader("Foot Measurement Analysis")
                st.warning("**Measurement Concerns Detected:**")
                for alert in measurement_alerts:
                    st.write(f"⚠️ {alert}")
                st.info("These measurements may indicate nerve damage or muscle changes associated with leprosy. Please discuss with a healthcare professional.")
            else:
                st.success("✅ Foot measurements appear normal and symmetrical")
            
            # Show measurement summary
            with st.expander("Measurement Summary", expanded=False):
                col_sum1, col_sum2 = st.columns(2)
                with col_sum1:
                    st.markdown("**Right Foot:**")
                    st.write(f"Length: {length_foot_right}cm")
                    st.write(f"Ball Girth: {ball_girth_right}cm")
                    st.write(f"In Step: {in_step_right}cm")
                with col_sum2:
                    st.markdown("**Left Foot:**")
                    st.write(f"Length: {length_foot_left}cm")
                    st.write(f"Ball Girth: {ball_girth_left}cm")
                    st.write(f"In Step: {in_step_left}cm")
                
                st.markdown("**Asymmetry Analysis:**")
                st.write(f"Length difference: {length_diff:.1f}cm")
                st.write(f"Ball girth difference: {ball_girth_diff:.1f}cm")
                st.write(f"In-step difference: {in_step_diff:.1f}cm")
            
            st.info("**Important:** This is a screening tool only. Always consult healthcare professionals for proper diagnosis.")

if __name__ == "__main__":
    main()