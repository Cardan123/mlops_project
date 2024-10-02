from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset

def generate_report(reference_data, current_data):
    report = Report(metrics=[
    DataDriftPreset(),
    ClassificationPreset()
    ])
    report.run(reference_data=reference_data, current_data=current_data)
    report.save_html("evidently_report.html")