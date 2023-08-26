### VideoRotation estimation using RotNet cnn, ref: https://github.com/d4nst/RotNet <br>


## Setup: <br>

```
python3 -m pip install .
```
## Pre-requirements: <br>
ffmpeg is required <br>


## Usage: <br>
```
from videorotation import VideoRotationAnalysis
with VideoRotationAnalysis(frames_per_video=12) as analysis:
    result, angles = analysis.check_if_upsidedown_for_video(video_path)
    print((result,angles))
```
