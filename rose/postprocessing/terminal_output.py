from typing import List, Dict, Any


class TerminalOutput:
    """
    Handles formatted terminal output for various processing results.
    """

    @staticmethod
    def print_object_detection_results(detections: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
        """
        Print object detection results to the terminal in a formatted way.

        Args:
            detections (List[Dict[str, Any]]): List of detected objects with their properties
            summary (Dict[str, Any]): Summary statistics of detections
        """
        print("\n" + "="*60)
        print("OBJECT DETECTION RESULTS")
        print("="*60)

        # Print summary
        print(f"Total objects detected: {summary['total_objects']}")
        print(f"Average confidence: {summary['average_confidence']:.3f}")
        print(f"Unique classes: {', '.join(summary['unique_classes'])}")

        # Print class counts
        if summary['class_counts']:
            print("\nClass distribution:")
            for class_name, count in summary['class_counts'].items():
                print(f"  {class_name}: {count}")

        # Print detailed detections
        if detections:
            print(f"\nDetailed detections ({len(detections)} objects):")
            print("-" * 60)
            for i, detection in enumerate(detections, 1):
                bbox = detection['bbox']
                print(f"{i:2d}. {detection['class_name']:15s} | "
                      f"Confidence: {detection['confidence']:.3f} | "
                      f"BBox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
        else:
            print("\nNo objects detected.")

        print("="*60)
