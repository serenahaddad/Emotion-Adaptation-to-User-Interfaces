#Author: Syrine HADDAD
from moviepy.video.io.VideoFileClip import VideoFileClip

def trim_video(input_path, output_path, start_time, end_time):
    # Load the video clip
    video = VideoFileClip(input_path)

    # Calculate the duration of the video
    video_duration = video.duration

    # Calculate the new duration after trimming
    new_duration = video_duration - (start_time + (video_duration - end_time))

    # Trim the video
    trimmed_video = video.subclip(start_time, end_time)

    # Write the trimmed video to a new file
    trimmed_video.write_videofile(output_path, codec='libx264', fps=video.fps)

    # Close the video file
    video.close()

# Input video path
input_path = "./*.mp4"

# Output video path
output_path = "./*.mp4"

# Trim the video 
start_time = X * Y + Z
end_time = X * Y + Z

# Call the function to trim the video
trim_video(input_path, output_path, start_time, end_time)
