import supervision as sv

class CrowdAnalytics:
    def __init__(self):
        self.reset_annotators()

    def reset_annotators(self):
        """Clears memory to handle different video resolutions (720p/1080p)."""
        self.heatmap_annotator = sv.HeatMapAnnotator()
        try:
            self.box_annotator = sv.BoxAnnotator()
        except AttributeError:
            self.box_annotator = sv.BoundingBoxAnnotator()

    def get_density_info(self, count, low_limit, high_limit):
        """Logic for user-defined crowd ranges."""
        if count <= low_limit:
            return "Low", "green"
        elif count <= high_limit:
            return "Moderate", "orange"
        else:
            return "High", "red"

    def annotate_frame(self, frame, detections):
        annotated_frame = self.heatmap_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=detections)
        return annotated_frame