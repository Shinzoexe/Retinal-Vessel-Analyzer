import os
import cv2
import tempfile
import streamlit as st

from utils.preprocessing import load_image, preprocess_fundus
from utils.segmentation import load_model, segment_vessels
from utils.features import extract_features
from utils.prediction import calculate_risk


# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="AI Retinal Stroke Risk Analysis",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 AI-Based Retinal Stroke Risk Analysis System")
st.markdown(
"""
This application automatically segments retinal blood vessels,
extracts vascular biomarkers, and estimates stroke risk using
a deep learning model.
"""
)

st.markdown("---")


# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------

st.sidebar.header("Project Information")

st.sidebar.info(
"""
**Model**

U-Net (ResNet34 Encoder)

**Input**

Retinal Fundus Image

**Output**

• Vessel Segmentation

• Retinal Features

• Stroke Risk Assessment
"""
)


# ---------------------------------------------------
# Load Deep Learning Model
# ---------------------------------------------------

MODEL_PATH = os.path.join(
    "models",
    "unet_vessel_segmentation.pth"
)


@st.cache_resource
def initialize_model():

    model = load_model(MODEL_PATH)

    return model


try:

    model = initialize_model()

except Exception as e:

    st.error(f"Unable to load model.\n\n{e}")

    st.stop()


# ---------------------------------------------------
# Upload Image
# ---------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Retinal Fundus Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".png"
    )

    temp_file.write(uploaded_file.read())

    temp_file.close()


    # ----------------------------
    # Preprocessing
    # ----------------------------

    original = load_image(temp_file.name)
    processed = preprocess_fundus(original)


    # ----------------------------
    # Vessel Segmentation
    # ----------------------------

    with st.spinner("Performing Vessel Segmentation..."):

        vessel_mask = segment_vessels(
            processed,
            model
        )


    # ----------------------------
    # Feature Extraction
    # ----------------------------

    features = extract_features(
        vessel_mask
    )


    # ----------------------------
    # Stroke Risk Prediction
    # ----------------------------

    score, category = calculate_risk(
        features
    )


    score_percent = score * 100


    st.success("Analysis Completed Successfully")

    st.markdown("---")


    col1, col2 = st.columns(2)


    with col1:

        st.subheader("Original Retinal Image")

        st.image(
            cv2.cvtColor(
                original,
                cv2.COLOR_BGR2RGB
            ),
            use_container_width=True
        )


    with col2:

        st.subheader("Segmented Blood Vessels")

        st.image(
            vessel_mask,
            clamp=True,
            use_container_width=True
        )

    st.markdown("---")
        # ----------------------------------------
    # Display Extracted Features
    # ----------------------------------------

    st.subheader("Extracted Retinal Features")

    feature_table = {
        "Feature": [
            "Vessel Density",
            "Vessel Length",
            "Branch Points",
            "End Points",
            "Tortuosity"
        ],
        "Value": [
            round(features["vessel_density"], 4),
            int(features["vessel_length"]),
            int(features["branch_points"]),
            int(features["end_points"]),
            round(features["tortuosity"], 4)
        ]
    }

    st.table(feature_table)

    st.markdown("---")

    # ----------------------------------------
    # Stroke Risk Result
    # ----------------------------------------

    st.subheader("Stroke Risk Assessment")

    if category == "Low":
        color = "🟢"
        recommendation = """
**Low Risk**

No significant vascular abnormalities detected.

Recommendation:
- Maintain a healthy lifestyle.
- Routine eye examination every 12 months.
"""
    elif category == "Medium":
        color = "🟡"
        recommendation = """
**Medium Risk**

Moderate retinal vascular abnormalities detected.

Recommendation:
- Consult an ophthalmologist.
- Monitor blood pressure and blood sugar.
- Consider neurological screening.
"""
    else:
        color = "🔴"
        recommendation = """
**High Risk**

Significant retinal vascular abnormalities detected.

Recommendation:
- Immediate clinical evaluation is recommended.
- Consult a neurologist and ophthalmologist.
- Additional cardiovascular investigations are advised.
"""

    metric1, metric2 = st.columns(2)

    with metric1:
        st.metric(
            label="Risk Score",
            value=f"{score_percent:.2f}%"
        )

    with metric2:
        st.metric(
            label="Risk Category",
            value=f"{color} {category}"
        )

    st.progress(score)

    st.markdown(recommendation)

    st.markdown("---")

    # ----------------------------------------
    # Download Results
    # ----------------------------------------

    report = f"""
AI RETINAL STROKE RISK ANALYSIS REPORT

Stroke Risk Score : {score_percent:.2f} %

Risk Category     : {category}

----------------------------------------

Retinal Features

Vessel Density : {features['vessel_density']:.4f}

Vessel Length  : {features['vessel_length']}

Branch Points  : {features['branch_points']}

End Points     : {features['end_points']}

Tortuosity     : {features['tortuosity']:.4f}

----------------------------------------

This report is generated automatically using
the AI-based retinal vessel analysis system.

For research and educational purposes only.
"""

    st.download_button(
        label="📄 Download Analysis Report",
        data=report,
        file_name="stroke_risk_report.txt",
        mime="text/plain"
    )

else:

    st.info("Please upload a retinal fundus image (JPG, JPEG, or PNG) to begin the analysis.")

st.markdown("---")

st.caption(
    "AI-Based Retinal Stroke Risk Analysis System | MSME Hackathon Project"
)