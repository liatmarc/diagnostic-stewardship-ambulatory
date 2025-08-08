# Ambulatory Diagnostic Stewardship – Static Dashboard (Simulated)

This package is ready for **GitHub Pages** and includes:
- **Overview** (classification charts + mapping)
- **Follow-up** (time-to-follow-up metrics for abnormal results)
- **Outreach** (tests not completed by department + outreach list)

## How to publish
1. Create a public repo named **diagnostic-stewardship-ambulatory**.
2. Upload **all files in this folder** to the repo root.
3. Go to **Settings → Pages**, choose **Deploy from a branch**, select the **main** branch and **/(root)** folder, then **Save**.
4. Your site will be live at the URL GitHub shows. The entry page is **index_amb.html**.

## Files
- `index_amb.html` – Tabbed dashboard (Overview, Follow-up, Outreach)
- `ambulatory_tests_raw_amb.csv`, `ambulatory_tests_processed_amb.csv`
- `summary_by_department_amb.csv`, `monthly_non_actionable_rate_amb.csv`
- `classification_interventions_amb.csv`, `data_quality_report_amb.csv`
- `follow_up_time_summary_amb.csv`, `follow_up_time_detail_amb.csv`
- `outreach_counts_by_department_amb.csv`, `outreach_list_amb.csv`
- `dept_classification_stacked_amb.png`, `monthly_non_actionable_rate_amb.png`
- `follow_up_time_distribution_amb.png`, `incomplete_by_department_amb.png`
