import cv2
from src.detector import CrowdDetector
from src.analytics import CrowdAnalytics

def run_local(path):
    detector = CrowdDetector()
    analytics = CrowdAnalytics()
    
    cap = cv2.VideoCapture(path)
    analytics.reset_annotators()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        detections = detector.get_detections(frame)
        output = analytics.annotate_frame(frame, detections)

        cv2.imshow("Press Q to Quit", output)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_local("data/sample.mp4")