import argparse
from videorotation import VideoRotationAnalysis

def main(video_path):
    with VideoRotationAnalysis(frames_per_video=12, frames_threshold=8) as analysis:
        result = analysis.check_if_upsidedown_for_video(video_path)
        print(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze video rotation.')
    parser.add_argument('video_path', type=str, help='Path to the video file.')

    args = parser.parse_args()
    main(args.video_path)
