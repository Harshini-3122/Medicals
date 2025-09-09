import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import re
from datetime import datetime, date
from typing import List, Dict, Optional, Tuple
import sqlite3
import hashlib
from dataclasses import dataclass
from enum import Enum
import plotly.express as px
import plotly.graph_objects as go
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore")

# Configure Streamlit page
st.set_page_config(
    page_title="AI Medical Prescription Verification",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .drug-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
    .warning-card {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .danger-card {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-card {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Data Models
@dataclass
class Drug:
    name: str
    dosage: str
    frequency: str
    route: str = "oral"
    generic_name: str = ""
    drug_class: str = ""
    indications: List[str] = None
    contraindications: List[str] = None
    side_effects: List[str] = None
    
    def __post_init__(self):
        if self.indications is None:
            self.indications = []
        if self.contraindications is None:
            self.contraindications = []
        if self.side_effects is None:
            self.side_effects = []

@dataclass
class Patient:
    age: int
    weight: Optional[float] = None
    height: Optional[float] = None
    gender: str = "Unknown"
    medical_conditions: List[str] = None
    allergies: List[str] = None
    
    def __post_init__(self):
        if self.medical_conditions is None:
            self.medical_conditions = []
        if self.allergies is None:
            self.allergies = []

class InteractionSeverity(Enum):
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    SEVERE = "Severe"

class DrugDatabase:
    """Simulated drug database with common medications"""
    
    def __init__(self):
        self.drugs_data = {
            "aspirin": {
                "generic_name": "acetylsalicylic acid",
                "drug_class": "NSAID",
                "indications": ["pain relief", "fever reduction", "cardiovascular protection"],
                "contraindications": ["bleeding disorders", "peptic ulcer", "severe liver disease"],
                "side_effects": ["stomach irritation", "bleeding", "tinnitus"],
                "max_daily_dose": {"adult": "4000mg", "pediatric": "100mg/kg", "elderly": "3000mg"},
                "interactions": ["warfarin", "methotrexate", "alcohol"]
            },
            "ibuprofen": {
                "generic_name": "ibuprofen",
                "drug_class": "NSAID",
                "indications": ["pain relief", "inflammation", "fever reduction"],
                "contraindications": ["kidney disease", "heart failure", "peptic ulcer"],
                "side_effects": ["stomach upset", "kidney problems", "high blood pressure"],
                "max_daily_dose": {"adult": "3200mg", "pediatric": "40mg/kg", "elderly": "2400mg"},
                "interactions": ["warfarin", "lithium", "ace inhibitors"]
            },
            "warfarin": {
                "generic_name": "warfarin sodium",
                "drug_class": "Anticoagulant",
                "indications": ["blood clot prevention", "atrial fibrillation", "valve replacement"],
                "contraindications": ["active bleeding", "pregnancy", "severe liver disease"],
                "side_effects": ["bleeding", "bruising", "hair loss"],
                "max_daily_dose": {"adult": "10mg", "elderly": "5mg"},
                "interactions": ["aspirin", "alcohol", "antibiotics", "vitamin k"]
            },
            "metformin": {
                "generic_name": "metformin hydrochloride",
                "drug_class": "Antidiabetic",
                "indications": ["type 2 diabetes", "insulin resistance", "PCOS"],
                "contraindications": ["kidney disease", "heart failure", "acidosis"],
                "side_effects": ["nausea", "diarrhea", "metallic taste"],
                "max_daily_dose": {"adult": "2000mg", "elderly": "1500mg"},
                "interactions": ["contrast dyes", "alcohol", "cimetidine"]
            },
            "lisinopril": {
                "generic_name": "lisinopril",
                "drug_class": "ACE Inhibitor",
                "indications": ["high blood pressure", "heart failure", "kidney protection"],
                "contraindications": ["pregnancy", "angioedema", "bilateral renal stenosis"],
                "side_effects": ["dry cough", "high potassium", "dizziness"],
                "max_daily_dose": {"adult": "40mg", "elderly": "20mg"},
                "interactions": ["potassium supplements", "nsaids", "lithium"]
            },
            "amlodipine": {
                "generic_name": "amlodipine besylate",
                "drug_class": "Calcium Channel Blocker",
                "indications": ["high blood pressure", "angina", "coronary artery disease"],
                "contraindications": ["severe aortic stenosis", "cardiogenic shock"],
                "side_effects": ["ankle swelling", "flushing", "palpitations"],
                "max_daily_dose": {"adult": "10mg", "elderly": "5mg"},
                "interactions": ["simvastatin", "grapefruit juice", "cyclosporine"]
            }
        }
    
    def get_drug_info(self, drug_name: str) -> Dict:
        """Get drug information from database"""
        drug_name = drug_name.lower().strip()
        return self.drugs_data.get(drug_name, {})
    
    def check_interaction(self, drug1: str, drug2: str) -> Tuple[bool, str, InteractionSeverity]:
        """Check for drug interactions between two drugs"""
        drug1_info = self.get_drug_info(drug1)
        drug2_info = self.get_drug_info(drug2)
        
        if not drug1_info or not drug2_info:
            return False, "Drug information not available", InteractionSeverity.LOW
        
        drug1_interactions = drug1_info.get("interactions", [])
        drug2_interactions = drug2_info.get("interactions", [])
        
        # Check if drugs interact
        if drug2.lower() in drug1_interactions or drug1.lower() in drug2_interactions:
            # Determine severity based on drug classes
            severity = self._determine_interaction_severity(drug1_info, drug2_info)
            interaction_desc = self._get_interaction_description(drug1, drug2, drug1_info, drug2_info)
            return True, interaction_desc, severity
        
        return False, "No known interactions", InteractionSeverity.LOW
    
    def _determine_interaction_severity(self, drug1_info: Dict, drug2_info: Dict) -> InteractionSeverity:
        """Determine interaction severity based on drug classes"""
        high_risk_combinations = [
            ("NSAID", "Anticoagulant"),
            ("ACE Inhibitor", "NSAID"),
            ("Anticoagulant", "NSAID")
        ]
        
        drug1_class = drug1_info.get("drug_class", "")
        drug2_class = drug2_info.get("drug_class", "")
        
        for combo in high_risk_combinations:
            if (drug1_class in combo and drug2_class in combo) or \
               (drug2_class in combo and drug1_class in combo):
                return InteractionSeverity.HIGH
        
        return InteractionSeverity.MODERATE
    
    def _get_interaction_description(self, drug1: str, drug2: str, drug1_info: Dict, drug2_info: Dict) -> str:
        """Get description of drug interaction"""
        interactions_desc = {
            ("aspirin", "warfarin"): "Increased bleeding risk. Monitor INR closely and adjust warfarin dose.",
            ("ibuprofen", "warfarin"): "Increased bleeding risk. Consider alternative pain relief.",
            ("ibuprofen", "lisinopril"): "Reduced effectiveness of blood pressure medication. Monitor BP.",
            ("aspirin", "ibuprofen"): "Increased GI bleeding risk. Take with food and monitor for symptoms."
        }
        
        key1 = (drug1.lower(), drug2.lower())
        key2 = (drug2.lower(), drug1.lower())
        
        return interactions_desc.get(key1, interactions_desc.get(key2, 
            f"Potential interaction between {drug1} and {drug2}. Consult healthcare provider."))

class GraniteNLPProcessor:
    """IBM Granite-based NLP processor for drug information extraction"""
    
    def __init__(self):
        # Initialize the pipeline (in real implementation, use IBM Granite models)
        try:
            # Placeholder for IBM Granite model
            # In production, replace with actual IBM Granite model
            self.nlp_pipeline = pipeline("text-classification", 
                                       model="microsoft/DialoGPT-medium",
                                       return_all_scores=True)
        except Exception as e:
            st.warning(f"NLP model not available: {e}")
            self.nlp_pipeline = None
    
    def extract_drug_info(self, text: str) -> List[Drug]:
        """Extract drug information from unstructured text"""
        drugs = []
        
        # Enhanced regex patterns for drug extraction
        drug_patterns = [
            r'(\w+)\s+(\d+(?:\.\d+)?)\s*(mg|g|ml|mcg|units?)\s+(?:(\d+)\s*times?\s*(?:daily|day|per day)|(?:every|q)\s*(\d+)\s*(?:hours?|hrs?|h))',
            r'(\w+)\s+(\d+(?:\.\d+)?)\s*(mg|g|ml|mcg|units?)\s+(daily|twice daily|three times daily|QID|BID|TID|PRN)',
            r'(\w+)\s+(\d+(?:\.\d+)?)\s*(mg|g|ml|mcg|units?)',
        ]
        
        for pattern in drug_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                drug_name = groups[0].lower()
                dosage = f"{groups[1]}{groups[2]}" if len(groups) > 2 else groups[1]
                
                # Determine frequency
                frequency = "once daily"  # default
                if len(groups) > 3 and groups[3]:
                    frequency = groups[3].lower()
                elif len(groups) > 4 and groups[4]:
                    freq_hours = int(groups[4])
                    if freq_hours <= 6:
                        frequency = "four times daily"
                    elif freq_hours <= 8:
                        frequency = "three times daily"
                    elif freq_hours <= 12:
                        frequency = "twice daily"
                    else:
                        frequency = "once daily"
                
                # Check if it's a known drug
                db = DrugDatabase()
                drug_info = db.get_drug_info(drug_name)
                
                if drug_info or self._is_likely_drug_name(drug_name):
                    drug = Drug(
                        name=drug_name,
                        dosage=dosage,
                        frequency=frequency,
                        generic_name=drug_info.get("generic_name", ""),
                        drug_class=drug_info.get("drug_class", ""),
                        indications=drug_info.get("indications", []),
                        contraindications=drug_info.get("contraindications", []),
                        side_effects=drug_info.get("side_effects", [])
                    )
                    drugs.append(drug)
        
        return list({drug.name: drug for drug in drugs}.values())  # Remove duplicates
    
    def _is_likely_drug_name(self, name: str) -> bool:
        """Check if a string is likely a drug name"""
        # Common drug name patterns and endings
        drug_endings = ['in', 'ol', 'ide', 'ine', 'ate', 'il', 'an', 'ox', 'ex']
        common_drugs = ['aspirin', 'tylenol', 'advil', 'motrin', 'aleve']
        
        name_lower = name.lower()
        
        # Check against common drugs
        if name_lower in common_drugs:
            return True
        
        # Check drug-like endings
        if any(name_lower.endswith(ending) for ending in drug_endings):
            return True
        
        # Check length and character patterns
        if 4 <= len(name) <= 15 and name.isalpha():
            return True
        
        return False

class DosageCalculator:
    """Calculate age-specific dosages"""
    
    @staticmethod
    def calculate_pediatric_dosage(adult_dose: str, patient_age: int, patient_weight: Optional[float] = None) -> str:
        """Calculate pediatric dosage based on age and weight"""
        try:
            # Extract numeric value from adult dose
            dose_match = re.search(r'(\d+(?:\.\d+)?)', adult_dose)
            if not dose_match:
                return "Consult pediatric dosing guidelines"
            
            adult_dose_value = float(dose_match.group(1))
            unit = re.search(r'[a-zA-Z]+', adult_dose).group() if re.search(r'[a-zA-Z]+', adult_dose) else "mg"
            
            # Age-based dosing (simplified)
            if patient_age < 2:
                pediatric_dose = adult_dose_value * 0.1
            elif patient_age < 6:
                pediatric_dose = adult_dose_value * 0.25
            elif patient_age < 12:
                pediatric_dose = adult_dose_value * 0.5
            else:
                pediatric_dose = adult_dose_value * 0.75
            
            return f"{pediatric_dose:.1f}{unit}"
        
        except Exception:
            return "Consult pediatric dosing guidelines"
    
    @staticmethod
    def calculate_elderly_dosage(adult_dose: str) -> str:
        """Calculate elderly dosage (typically reduced)"""
        try:
            dose_match = re.search(r'(\d+(?:\.\d+)?)', adult_dose)
            if not dose_match:
                return "Consult geriatric dosing guidelines"
            
            adult_dose_value = float(dose_match.group(1))
            unit = re.search(r'[a-zA-Z]+', adult_dose).group() if re.search(r'[a-zA-Z]+', adult_dose) else "mg"
            
            # Reduce dose for elderly (typically 25-50% reduction)
            elderly_dose = adult_dose_value * 0.75
            
            return f"{elderly_dose:.1f}{unit}"
        
        except Exception:
            return "Consult geriatric dosing guidelines"

class AlternativeMedicationFinder:
    """Find alternative medications for drug interactions or contraindications"""
    
    def __init__(self):
        self.alternatives = {
            "aspirin": ["acetaminophen", "ibuprofen", "naproxen"],
            "ibuprofen": ["acetaminophen", "naproxen", "aspirin"],
            "warfarin": ["apixaban", "rivaroxaban", "dabigatran"],
            "metformin": ["glyburide", "glipizide", "insulin"],
            "lisinopril": ["losartan", "amlodipine", "metoprolol"],
            "amlodipine": ["lisinopril", "metoprolol", "hydrochlorothiazide"]
        }
        
        self.alternative_info = {
            "acetaminophen": "Safer for patients with bleeding disorders or GI issues",
            "naproxen": "Longer-acting NSAID, less frequent dosing",
            "apixaban": "Direct oral anticoagulant, less monitoring required",
            "losartan": "ARB with similar effects to ACE inhibitors but less cough",
            "metoprolol": "Beta-blocker for blood pressure and heart rate control"
        }
    
    def find_alternatives(self, drug_name: str, reason: str = "") -> List[Dict[str, str]]:
        """Find alternative medications"""
        alternatives = []
        drug_name = drug_name.lower()
        
        if drug_name in self.alternatives:
            for alt in self.alternatives[drug_name]:
                alternatives.append({
                    "name": alt,
                    "reason": self.alternative_info.get(alt, "Alternative medication"),
                    "note": f"Consider as alternative to {drug_name}"
                })
        
        return alternatives

class PrescriptionVerifier:
    """Main prescription verification system"""
    
    def __init__(self):
        self.drug_db = DrugDatabase()
        self.nlp_processor = GraniteNLPProcessor()
        self.dosage_calculator = DosageCalculator()
        self.alt_finder = AlternativeMedicationFinder()
    
    def verify_prescription(self, drugs: List[Drug], patient: Patient) -> Dict:
        """Comprehensive prescription verification"""
        results = {
            "interactions": [],
            "dosage_warnings": [],
            "contraindications": [],
            "alternatives": [],
            "recommendations": []
        }
        
        # Check drug interactions
        for i, drug1 in enumerate(drugs):
            for drug2 in drugs[i+1:]:
                has_interaction, desc, severity = self.drug_db.check_interaction(drug1.name, drug2.name)
                if has_interaction:
                    results["interactions"].append({
                        "drug1": drug1.name,
                        "drug2": drug2.name,
                        "severity": severity.value,
                        "description": desc
                    })
        
        # Check dosages and contraindications
        for drug in drugs:
            drug_info = self.drug_db.get_drug_info(drug.name)
            
            # Age-specific dosage check
            if patient.age < 18:
                recommended_dose = self.dosage_calculator.calculate_pediatric_dosage(
                    drug.dosage, patient.age, patient.weight
                )
                if recommended_dose != drug.dosage:
                    results["dosage_warnings"].append({
                        "drug": drug.name,
                        "current_dose": drug.dosage,
                        "recommended_dose": recommended_dose,
                        "reason": "Pediatric dosing adjustment needed"
                    })
            elif patient.age >= 65:
                recommended_dose = self.dosage_calculator.calculate_elderly_dosage(drug.dosage)
                results["dosage_warnings"].append({
                    "drug": drug.name,
                    "current_dose": drug.dosage,
                    "recommended_dose": recommended_dose,
                    "reason": "Consider elderly dosing adjustment"
                })
            
            # Check contraindications
            contraindications = drug_info.get("contraindications", [])
            for condition in patient.medical_conditions:
                if any(contra.lower() in condition.lower() for contra in contraindications):
                    results["contraindications"].append({
                        "drug": drug.name,
                        "condition": condition,
                        "warning": f"{drug.name} is contraindicated in {condition}"
                    })
        
        # Generate recommendations
        self._generate_recommendations(results, drugs, patient)
        
        return results
    
    def _generate_recommendations(self, results: Dict, drugs: List[Drug], patient: Patient):
        """Generate clinical recommendations"""
        recommendations = []
        
        # Interaction recommendations
        for interaction in results["interactions"]:
            if interaction["severity"] in ["High", "Severe"]:
                alternatives = self.alt_finder.find_alternatives(interaction["drug1"])
                if alternatives:
                    recommendations.append(f"Consider replacing {interaction['drug1']} with {alternatives[0]['name']}")
        
        # Age-specific recommendations
        if patient.age >= 65:
            recommendations.append("Regular monitoring recommended for elderly patient")
            recommendations.append("Consider lower starting doses and slower titration")
        
        if patient.age < 18:
            recommendations.append("Verify pediatric dosing and safety profile")
        
        results["recommendations"] = recommendations

# Initialize session state
if "verification_results" not in st.session_state:
    st.session_state.verification_results = None
if "extracted_drugs" not in st.session_state:
    st.session_state.extracted_drugs = []

# Main Application
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 AI Medical Prescription Verification System</h1>
        <p>Powered by IBM Granite & Advanced NLP | Drug Interaction Detection & Safety Analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "Prescription Analysis",
        "Drug Database",
        "Interaction Checker",
        "Dosage Calculator",
        "Alternative Finder"
    ])
    
    if page == "Prescription Analysis":
        prescription_analysis_page()
    elif page == "Drug Database":
        drug_database_page()
    elif page == "Interaction Checker":
        interaction_checker_page()
    elif page == "Dosage Calculator":
        dosage_calculator_page()
    elif page == "Alternative Finder":
        alternative_finder_page()

def prescription_analysis_page():
    st.header("📋 Comprehensive Prescription Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Patient Information")
        
        # Patient details
        col1a, col1b, col1c = st.columns(3)
        with col1a:
            age = st.number_input("Age", min_value=0, max_value=120, value=45)
        with col1b:
            weight = st.number_input("Weight (kg)", min_value=0.0, max_value=300.0, value=70.0, step=0.1)
        with col1c:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        
        # Medical conditions and allergies
        conditions = st.text_input("Medical Conditions (comma-separated)", 
                                  placeholder="diabetes, hypertension, heart disease")
        allergies = st.text_input("Known Allergies (comma-separated)", 
                                placeholder="penicillin, sulfa, aspirin")
        
        st.subheader("Prescription Input")
        input_method = st.radio("Input Method", ["Manual Entry", "Text Analysis"])
        
        if input_method == "Manual Entry":
            manual_drug_entry()
        else:
            text_analysis_entry()
    
    with col2:
        st.subheader("Analysis Status")
        if st.session_state.extracted_drugs:
            st.success(f"✅ {len(st.session_state.extracted_drugs)} drugs identified")
            for drug in st.session_state.extracted_drugs:
                st.info(f"**{drug.name.title()}** - {drug.dosage} {drug.frequency}")
        else:
            st.warning("⚠️ No drugs entered yet")
        
        if st.button("🔍 Analyze Prescription", type="primary", use_container_width=True):
            if st.session_state.extracted_drugs:
                analyze_prescription(age, weight, gender, conditions, allergies)
            else:
                st.error("Please enter prescription information first")
    
    # Display results
    if st.session_state.verification_results:
        display_analysis_results()

def manual_drug_entry():
    st.write("**Add Medications Manually**")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        drug_name = st.text_input("Drug Name", key="manual_drug_name")
    with col2:
        dosage = st.text_input("Dosage (e.g., 500mg)", key="manual_dosage")
    with col3:
        frequency = st.selectbox("Frequency", 
                                ["Once daily", "Twice daily", "Three times daily", 
                                 "Four times daily", "As needed"], key="manual_frequency")
    
    if st.button("Add Drug"):
        if drug_name and dosage:
            drug = Drug(name=drug_name.lower(), dosage=dosage, frequency=frequency)
            st.session_state.extracted_drugs.append(drug)
            st.success(f"Added {drug_name}")
            st.rerun()
    
    # Display current drugs
    if st.session_state.extracted_drugs:
        st.write("**Current Medications:**")
        for i, drug in enumerate(st.session_state.extracted_drugs):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"{i+1}. **{drug.name.title()}** - {drug.dosage} {drug.frequency}")
            with col2:
                if st.button("Remove", key=f"remove_{i}"):
                    st.session_state.extracted_drugs.pop(i)
                    st.rerun()

def text_analysis_entry():
    st.write("**Extract from Medical Text**")
    
    prescription_text = st.text_area(
        "Enter prescription or medical text:",
        height=150,
        placeholder="""Example:
Patient prescribed aspirin 81mg once daily for cardiovascular protection.
Also taking ibuprofen 400mg twice daily for joint pain.
Warfarin 5mg daily for atrial fibrillation."""
    )
    
    if st.button("🧠 Extract Drug Information"):
        if prescription_text:
            processor = GraniteNLPProcessor()
            extracted_drugs = processor.extract_drug_info(prescription_text)
            
            if extracted_drugs:
                st.session_state.extracted_drugs = extracted_drugs
                st.success(f"Extracted {len(extracted_drugs)} medications!")
                st.rerun()
            else:
                st.warning("No medications found in the text. Try manual entry or check the text format.")
        else:
            st.error("Please enter prescription text")

def analyze_prescription(age, weight, gender, conditions, allergies):
    # Create patient object
    patient = Patient(
        age=age,
        weight=weight,
        gender=gender,
        medical_conditions=[c.strip() for c in conditions.split(",") if c.strip()],
        allergies=[a.strip() for a in allergies.split(",") if a.strip()]
    )
    
    # Verify prescription
    verifier = PrescriptionVerifier()
    results = verifier.verify_prescription(st.session_state.extracted_drugs, patient)
    
    st.session_state.verification_results = results

def display_analysis_results():
    results = st.session_state.verification_results
    
    st.header("📊 Analysis Results")
    
    # Create metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Drug Interactions", len(results["interactions"]))
    with col2:
        st.metric("Dosage Warnings", len(results["dosage_warnings"]))
    with col3:
        st.metric("Contraindications", len(results["contraindications"]))
    with col4:
        st.metric("Recommendations", len(results["recommendations"]))
    
    # Drug Interactions
    if results["interactions"]:
        st.subheader("⚠️ Drug Interactions")
        for interaction in results["interactions"]:
            severity_color = {
                "Low": "success-card",
                "Moderate": "warning-card", 
                "High": "danger-card",
                "Severe": "danger-card"
            }
            
            st.markdown(f"""
            <div class="{severity_color.get(interaction['severity'], 'warning-card')}">
                <h4>🔄 {interaction['drug1'].title()} ↔ {interaction['drug2'].title()}</h4>
                <p><strong>Severity:</strong> {interaction['severity']}</p>
                <p>{interaction['description']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Dosage Warnings
    if results["dosage_warnings"]:
        st.subheader("💊 Dosage Recommendations")
        for warning in results["dosage_warnings"]:
            st.markdown(f"""
            <div class="warning-card">
                <h4>📏 {warning['drug'].title()}</h4>
                <p><strong>Current Dose:</strong> {warning['current_dose']}</p>
                <p><strong>Recommended Dose:</strong> {warning['recommended_dose']}</p>
                <p><strong>Reason:</strong> {warning['reason']}</p>
            </div>
            """, unsafe_allow_html=True)
    
def display_analysis_results():
    results = st.session_state.verification_results
    
    st.header("📊 Analysis Results")
    
    # Create metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Drug Interactions", len(results["interactions"]))
    with col2:
        st.metric("Dosage Warnings", len(results["dosage_warnings"]))
    with col3:
        st.metric("Contraindications", len(results["contraindications"]))
    with col4:
        st.metric("Recommendations", len(results["recommendations"]))
    
    # Drug Interactions
    if results["interactions"]:
        st.subheader("⚠️ Drug Interactions")
        for interaction in results["interactions"]:
            severity_color = {
                "Low": "success-card",
                "Moderate": "warning-card", 
                "High": "danger-card",
                "Severe": "danger-card"
            }
            
            st.markdown(f"""
            <div class="{severity_color.get(interaction['severity'], 'warning-card')}">
                <h4>🔄 {interaction['drug1'].title()} ↔ {interaction['drug2'].title()}</h4>
                <p><strong>Severity:</strong> {interaction['severity']}</p>
                <p>{interaction['description']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Dosage Warnings
    if results["dosage_warnings"]:
        st.subheader("💊 Dosage Recommendations")
        for warning in results["dosage_warnings"]:
            st.markdown(f"""
            <div class="warning-card">
                <h4>📏 {warning['drug'].title()}</h4>
                <p><strong>Current Dose:</strong> {warning['current_dose']}</p>
                <p><strong>Recommended Dose:</strong> {warning['recommended_dose']}</p>
                <p><strong>Reason:</strong> {warning['reason']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Recommendations
    if results["recommendations"]:
        st.subheader("✅ Recommendations")
        for rec in results["recommendations"]:
            st.markdown(f"""
            <div class="success-card">
                <p>{rec}</p>
            </div>
            """, unsafe_allow_html=True)
