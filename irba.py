import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

# ----------------- Load Data -----------------
folder_path = os.path.expanduser("~/Downloads/IRBA_Case_Study_Tables")

claim_basic = pd.read_csv(os.path.join(folder_path, "Claim_Basic.csv"))
claim_diagnosis = pd.read_csv(os.path.join(folder_path, "Claim_Diagnosis.csv"))
claim_doctor = pd.read_csv(os.path.join(folder_path, "Claim_Doctor.csv"))
doctor_info = pd.read_csv(os.path.join(folder_path, "Doctor_Info.csv"))
hospital_info = pd.read_csv(os.path.join(folder_path, "Hospital_Info.csv"))
policy_info = pd.read_csv(os.path.join(folder_path, "Policy_Info.csv"))

# ----------------- Preprocessing -----------------
claim_basic["Admission Date"] = pd.to_datetime(claim_basic["Admission Date"], errors="coerce")
claim_basic["Discharged Date"] = pd.to_datetime(claim_basic["Discharged Date"], errors="coerce")
claim_basic["Length_of_Stay"] = (claim_basic["Discharged Date"] - claim_basic["Admission Date"]).dt.days + 1
claim_basic.loc[claim_basic["Length_of_Stay"] < 0, "Length_of_Stay"] = None

policy_info["Inception Date"] = pd.to_datetime(policy_info["Inception Date"], errors="coerce")
policy_info["Age"] = policy_info["Inception Date"].dt.year - policy_info["Birth Year"]
policy_info.loc[policy_info["Age"] < 0, "Age"] = None

# ----------------- Standardize Doctor Specialty -----------------
doctor_info["Specialty"] = doctor_info["Specialty"].str.strip().str.title()
# Specifically ensure "Paediatrician" is consistent
doctor_info["Specialty"] = doctor_info["Specialty"].replace({
    "Paediatrician": "Paediatrician",
    "Pediatrician": "Paediatrician",
    "Paediatrics": "Paediatrician"
})

# ---- Merge Summaries ----
diagnosis_count = claim_diagnosis.groupby("Claim ID")["Diagnosis"].count().reset_index(name="Num_Diagnoses")
doctor_count = claim_doctor.groupby("Claim ID")["Doctor ID"].count().reset_index(name="Num_Doctors")

master = claim_basic.copy()
master = pd.merge(master, diagnosis_count, on="Claim ID", how="left")
master = pd.merge(master, doctor_count, on="Claim ID", how="left")
master = pd.merge(master, policy_info, on="Policy ID", how="left")
master = pd.merge(master, hospital_info, on="Hospital ID", how="left")

# ---- Rule-based Flags ----
master["flag_many_diagnoses"] = master.get("Num_Diagnoses", 0) > 3
master["flag_shortstay_manydiag"] = (master.get("Length_of_Stay", 0) <= 2) & (master.get("Num_Diagnoses", 0) > 2)
master["flag_longstay_onediag"] = (master.get("Length_of_Stay", 0) > 7) & (master.get("Num_Diagnoses", 0) == 1)
master["flag_manydoctors"] = master.get("Num_Doctors", 0) > 2
master["flag_age_missing"] = master["Age"].isna()

# Collect flagged IDs
rule_flags = [col for col in master.columns if col.startswith("flag_")]
claims_per_rule = {rule: master.loc[master[rule] == True, "Claim ID"].tolist() for rule in rule_flags}
hospitals_per_rule = {rule: master.loc[master[rule] == True, "Hospital ID"].unique().tolist() for rule in rule_flags}
doctors_per_rule = {rule: master.loc[master[rule] == True, "Num_Doctors"].tolist() for rule in rule_flags}
agents_per_rule = {rule: master.loc[master[rule] == True, "Agent"].unique().tolist() for rule in rule_flags if "Agent" in master.columns}


# ----------------- Streamlit Layout -----------------
st.set_page_config(page_title="IRBA Case Study", layout="wide")
st.title("🏥 Insurance Fraud / Waste / Abuse Analysis")

# Main Tabs
tabs = st.tabs(["EDA", "Rules"])

# ----------------- EDA Tab -----------------
with tabs[0]:
    st.header("Exploratory Data Analysis")
    eda_tabs = st.tabs(["Overall", "Claims", "Policy", "Hospital", "Doctor", "Agent"])

    # Overall
    with eda_tabs[0]:
        st.subheader("Overall Dataset Shapes")
        shapes = {
            "Claim_Basic": claim_basic.shape,
            "Claim_Diagnosis": claim_diagnosis.shape,
            "Claim_Doctor": claim_doctor.shape,
            "Doctor_Info": doctor_info.shape,
            "Hospital_Info": hospital_info.shape,
            "Policy_Info": policy_info.shape,
        }
        st.dataframe(pd.DataFrame(shapes, index=["rows", "cols"]))

        st.subheader("Summary Statistics - Claims")
        st.dataframe(claim_basic.describe(include="all"))

    # Claims
    with eda_tabs[1]:
        st.subheader("Claims Data Overview")
        st.dataframe(claim_basic.head())

        # Distribution of Length of Stay
        fig, ax = plt.subplots()
        claim_basic["Length_of_Stay"].dropna().hist(ax=ax, bins=20)
        ax.set_title("Distribution of Length of Stay")
        st.pyplot(fig)

        # Top 10 Diagnoses by frequency
        top_diagnoses = claim_diagnosis["Diagnosis"].value_counts().head(10)
        fig, ax = plt.subplots()
        top_diagnoses.plot(kind="bar", ax=ax)
        ax.set_title("Top 10 Diagnoses")
        st.pyplot(fig)

    # Policy
    with eda_tabs[2]:
        st.subheader("Policy Data Overview")
        st.dataframe(policy_info.head())

        gender_count = policy_info.groupby("Gender")["Policy ID"].count().reset_index()
        fig, ax = plt.subplots()
        ax.bar(gender_count["Gender"], gender_count["Policy ID"])
        ax.set_title("Gender Distribution")
        st.pyplot(fig)

        product_count = policy_info.groupby("Product")["Policy ID"].count().reset_index()
        fig, ax = plt.subplots()
        ax.bar(product_count["Product"], product_count["Policy ID"])
        ax.set_title("Product Distribution")
        st.pyplot(fig)

        # Age distribution
        fig, ax = plt.subplots()
        policy_info["Age"].dropna().hist(ax=ax, bins=20)
        ax.set_title("Age Distribution of Policyholders")
        st.pyplot(fig)

    # Hospital
    with eda_tabs[3]:
        st.subheader("Hospital Data Overview")
        st.dataframe(hospital_info.head())

        location_count = hospital_info.groupby("Location")["Hospital ID"].count().reset_index()
        fig, ax = plt.subplots()
        ax.bar(location_count["Location"], location_count["Hospital ID"])
        ax.set_title("Hospitals per Location")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # Doctor
    with eda_tabs[4]:
        st.subheader("Doctor Data Overview")
        st.dataframe(doctor_info.head())

        specialty_count = doctor_info.groupby("Specialty")["Doctor ID"].count().reset_index()
        fig, ax = plt.subplots()
        ax.bar(specialty_count["Specialty"], specialty_count["Doctor ID"])
        ax.set_title("Doctors per Specialty")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # Agent
    with eda_tabs[5]:
        st.subheader("Top Agents by Number of Policies")
        agent_count = policy_info.groupby("Agent")["Policy ID"].count().reset_index()
        agent_count = agent_count.sort_values("Policy ID", ascending=False).head(10)

        fig, ax = plt.subplots()
        ax.bar(agent_count["Agent"], agent_count["Policy ID"])
        ax.set_title("Top 10 Agents")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Distribution of policies per agent
        fig, ax = plt.subplots()
        policy_info.groupby("Agent")["Policy ID"].count().hist(ax=ax, bins=20)
        ax.set_title("Distribution of Policies per Agent")
        st.pyplot(fig)

# ================= Rules Tab =================
with tabs[1]:
    rule_tabs = st.tabs(["Claims", "Hospitals", "Doctors", "Agents"])

    # ---- Claim-level ----
    with rule_tabs[0]:
        st.subheader("Claim-level Rules")
        summary = pd.DataFrame({rule: [len(ids)] for rule, ids in claims_per_rule.items()}, index=["#Claims Flagged"])
        st.dataframe(summary)

        fig, ax = plt.subplots()
        ax.bar(claims_per_rule.keys(), [len(ids) for ids in claims_per_rule.values()])
        ax.set_title("Claims Flagged per Rule")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        for rule, ids in claims_per_rule.items():
            st.write(f"**{rule}**: {len(ids)} claims flagged")
            st.download_button(f"Download {rule} Claims", pd.DataFrame(ids, columns=["Claim ID"]).to_csv(index=False), file_name=f"{rule}_claims.csv")

    # ---- Hospital-level ----
    with rule_tabs[1]:
        st.subheader("Hospital-level Rules (Top by Flags)")
        # Count total flags per hospital
        hospital_flags = master.groupby("Hospital ID")[rule_flags].sum()
        hospital_flags["Total_Flags"] = hospital_flags.sum(axis=1)
        top_hospitals = hospital_flags.sort_values("Total_Flags", ascending=False).head(10)
        st.dataframe(top_hospitals)

        fig, ax = plt.subplots()
        ax.bar(top_hospitals.index.astype(str), top_hospitals["Total_Flags"], color="orange")
        ax.set_title("Top 10 Hospitals by Total Flags")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # ---- Doctor-level ----
    with rule_tabs[2]:
        st.subheader("Doctor-level Rules (Top by Flags)")
        # Sum flags per doctor using claim_doctor mapping
        doctor_flags = claim_doctor.merge(master[["Claim ID"] + rule_flags], on="Claim ID", how="left")
        doctor_flags_sum = doctor_flags.groupby("Doctor ID")[rule_flags].sum()
        doctor_flags_sum["Total_Flags"] = doctor_flags_sum.sum(axis=1)
        top_doctors = doctor_flags_sum.sort_values("Total_Flags", ascending=False).head(10)
        st.dataframe(top_doctors)

        fig, ax = plt.subplots()
        ax.bar(top_doctors.index.astype(str), top_doctors["Total_Flags"], color="green")
        ax.set_title("Top 10 Doctors by Total Flags")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # ---- Agent-level ----
    with rule_tabs[3]:
        st.subheader("Agent-level Rules (Top by Flags)")
        if "Agent" in master.columns:
            # Sum flags per agent
            agent_flags = master.groupby("Agent")[rule_flags].sum()
            agent_flags["Total_Flags"] = agent_flags.sum(axis=1)
            top_agents = agent_flags.sort_values("Total_Flags", ascending=False).head(10)
            st.dataframe(top_agents)

            fig, ax = plt.subplots()
            ax.bar(top_agents.index.astype(str), top_agents["Total_Flags"], color="purple")
            ax.set_title("Top 10 Agents by Total Flags")
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.info("No Agent data available in this dataset.")
