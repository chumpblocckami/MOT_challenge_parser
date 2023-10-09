import argparse

from src.people_tracking import PeopleTracker

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Database",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input_folder", help="Input folder", default="./data/frames/camera_2")
    parser.add_argument("-m", "--model", help="Model url",
                        default="https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt")
    parser.add_argument("-s", "--save_folder", help="Model save folder", default="./data/models")
    parser.add_argument("-o", "--output", help="output file", default="./")
    parser.add_argument("-fr", "--frame_rate", help="frame rate", default="30")
    parser.add_argument("-c", "--challenge", help="challenge rate", default="tmp")
    parser.add_argument("-f", "--format", help="frame format", default=".png")
    args = parser.parse_args()

    people_tracker = PeopleTracker(model_url=args.model,
                                   model_save_path=args.save_folder)
    people_tracker(input_folder=args.input_folder,
                   output_folder=args.output,
                   challenge_name=args.challenge,
                   frame_rate=args.frame_rate,
                   frame_format=args.format)
