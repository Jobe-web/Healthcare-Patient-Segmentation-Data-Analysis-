# ==========================================
# Import Libraries
# ==========================================

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import joblib
import os

css = """
.gradio-container{
    background:#F4F9FD;
}

h1{
    color:#0B6FA4;
    text-align:center;
    font-size:38px;
}

h2{
    color:#0B6FA4;
}

button{
    background:#0B6FA4 !important;
    color:white !important;
    border-radius:10px !important;
}

footer{
    display:none;
}
"""

# ==========================================
# Load Saved Files
# ==========================================

model = joblib.load("patient_segmentation_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")


# ==========================================
# Get dropdown values from the label encoders
# ==========================================


states = list(label_encoders["State"].classes_)
cities = list(label_encoders["City"].classes_)
insurance_types = list(label_encoders["Insurance_Type"].classes_)
conditions = list(label_encoders["Primary_Condition"].classes_)

print(states)
print(type(states))
print(cities)
print(type(cities))

# ==========================================
# Create the prediction function
# ==========================================

def predict_cluster(age, gender, state, city,
                    height, weight, bmi,
                    insurance, condition,
                    chronic, visits, billing,
                    days, preventive):

    gender = label_encoders["Gender"].transform([gender])[0]
    state = label_encoders["State"].transform([state])[0]
    city = label_encoders["City"].transform([city])[0]
    insurance = label_encoders["Insurance_Type"].transform([insurance])[0]
    condition = label_encoders["Primary_Condition"].transform([condition])[0]
    

    data = pd.DataFrame([[
      age, gender, state, city,
      height, weight, bmi,
      insurance, condition,
      chronic, visits, billing,
      days, preventive
    ]], columns=[
      "Age","Gender","State","City",
      "Height_cm","Weight_kg","BMI",
      "Insurance_Type","Primary_Condition",
      "Num_Chronic_Conditions",
      "Annual_Visits",
      "Avg_Billing_Amount",
      "Days_Since_Last_Visit",
      "Preventive_Care_Flag"
   ])
    data = scaler.transform(data)

    cluster = model.predict(data)[0]

    cluster_info = {
        0: {
            "title": "🟢 Healthy Lifestyle",
            "description": """
• Low healthcare visits
• Few chronic conditions
• Low medical costs
• Continue regular checkups.
"""
        },

        1: {
            "title": "🟡 Moderate Risk",
            "description": """
• Some chronic conditions
• Moderate healthcare costs
• Regular monitoring recommended.
"""
        },

        2: {
            "title": "🔴 High Risk",
            "description": """
• High healthcare utilization
• Multiple chronic conditions
• Needs frequent medical care.
"""
        }
    }

    # Create dashboard
    fig, axs = plt.subplots(2, 2, figsize=(8, 6))

    # -----------------------------
    # BMI
    # -----------------------------
    axs[0,0].bar(["BMI"], [bmi])
    axs[0,0].set_title("❤️ BMI")

    # -----------------------------
    # Annual Visits
    # -----------------------------
    axs[0,1].bar(["Visits"], [visits])
    axs[0,1].set_title("🏥 Annual Visits")

    # -----------------------------
    # Chronic Conditions
    # -----------------------------
    axs[1,0].bar(["Conditions"], [chronic])
    axs[1,0].set_title("💊 Chronic Conditions")

    # -----------------------------
    # Predicted Cluster
    # -----------------------------
    cluster_names = ["Healthy", "Moderate", "High"]
    colors = ["lightgray", "lightgray", "lightgray"]
    colors[cluster] = "#0B6FA4"
    axs[1,1].bar(cluster_names, [1, 1, 1], color=colors)
    axs[1,1].set_title("📊 Predicted Cluster")

    plt.tight_layout()
    dashboard = "dashboard.png"
    plt.savefig(dashboard)
    plt.close()
    chart = dashboard

    report = pd.DataFrame({
        "Patient Detail":[
            "Age",
            "Gender",
            "State",
            "City",
            "Insurance",
            "Condition",
            "Predicted Cluster"
        ],
        "Value":[
            age,
            gender,
            state,
            city,
            insurance,
            condition,
            cluster_info[cluster]["title"]
        ]
    })

    report_file = "Patient_Report.csv"
    report.to_csv(report_file, index=False)

    title = cluster_info[cluster]["title"]
    description = cluster_info[cluster]["description"]

    return title, description, dashboard, report_file


# ==========================================
# Build the Gradio interface
# ==========================================
with gr.Blocks(css=css) as demo:

    gr.Image(
      "banner.jpg",
       width=700,
       height=220,
       show_label=False,
       interactive=False,
       container=False
    )

    gr.Markdown(
      """
      # 🏥 Healthcare Patient Segmentation

      ### AI-Powered Patient Risk Clustering using Machine Learning (K-Means)

      Predict patient healthcare risk levels based on demographic, clinical, and healthcare utilization data.
      """
    )

    with gr.Row():

        #########################
        # LEFT SIDE
        #########################

        with gr.Column():

            age = gr.Number(label="Age")

            gender = gr.Dropdown(
                choices=["Male","Female"],
                label="Gender"
            )

            state = gr.Dropdown(
                choices=states,
                label="State"
            )

            city = gr.Dropdown(
                choices=cities,
                label="City"
            )

            height = gr.Number(label="Height (cm)")

            weight = gr.Number(label="Weight (kg)")

            bmi = gr.Number(label="BMI")

            insurance = gr.Dropdown(
                choices=insurance_types,
                label="Insurance"
            )

            condition = gr.Dropdown(
                choices=conditions,
                label="Condition"
            )

            chronic = gr.Number(label="Chronic Conditions")

            visits = gr.Number(label="Annual Visits")

            billing = gr.Number(label="Average Billing")

            days = gr.Number(label="Days Since Last Visit")

            preventive = gr.Dropdown(
                choices=[0,1],
                label="Preventive Care(Yes=1, No=0)"
            )

            predict_btn = gr.Button("🔍 Predict Cluster")

        #########################
        # RIGHT SIDE
        #########################

        with gr.Column():

            title = gr.Textbox(
                label="📊 Predicted Cluster"
            )

            description = gr.Textbox(
                label="📋 Health Summary",
                lines=8
            )

            chart = gr.Image(
                label="📈 Patient Dashboard"
            )

            report = gr.File(
                label="📥 Download Patient Report"
            )

    predict_btn.click(

        fn=predict_cluster,

        inputs=[
            age,
            gender,
            state,
            city,
            height,
            weight,
            bmi,
            insurance,
            condition,
            chronic,
            visits,
            billing,
            days,
            preventive
        ],

        outputs=[
            title,
            description,
            chart,
            report
        ]
    )

    gr.Markdown(
      """
      ---
        ### 👨‍💻 Developed by Lindokuhle Siphamandla Jobe

        Information Technology Student

        Machine Learning • Python • Gradio • K-Means Clustering
       """
   )

port = int(os.environ.get("PORT", 7860))
demo.launch(
server_name="0.0.0.0",
server_port=port
)