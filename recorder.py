#Author: Syrine HADDAD
import numpy as np
import cv2
import time
import argparse
import os
import pyautogui


parser = argparse.ArgumentParser(description='Record video from webcam and screen',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--webcam_number', default=1, type=int,
                    help='the number of webcam, if you have more than 1 webcam')
parser.add_argument('--fps', default=10, type=int,
                    help='number of frames per second for webcam recording')
parser.add_argument('--dir', default="",
                    help='directory to save the files (video, and log) in it')
parser.add_argument('--fid', default=f"{int(time.time() * 1000)}",
                    help='file name for the recorded video and log file')
parser.add_argument('--filename_prefix', default="",
                    help='prefix for the file names (video, and log files)')
parser.add_argument('--filename_postfix', default="REC",
                    help='postfix for the file names (video, and log files)')

args = parser.parse_args()


def record_video(filename, webcam_number, fps, dir=""):

    log_file_path = f"{filename}.log"
    webcam_video_file_path = f"{filename}_webcam.mp4"
    screen_video_file_path = f"{filename}_screen.mp4"

    if dir != "":
        if not os.path.exists(dir):
            os.makedirs(dir)
        log_file_path = os.path.join(dir, log_file_path)
        webcam_video_file_path = os.path.join(dir, webcam_video_file_path)
        screen_video_file_path = os.path.join(dir, screen_video_file_path)

    log_file = open(log_file_path, 'w')

    capture = cv2.VideoCapture(webcam_number)

    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    is_first_frame = True

    fourcc = cv2.VideoWriter_fourcc(*'H264')
    webcam_out = cv2.VideoWriter(webcam_video_file_path, fourcc, fps, size)
    screen_out = cv2.VideoWriter(screen_video_file_path, fourcc, fps, pyautogui.size())

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            print("Webcam Stream is not available!")
            break

        # write the webcam frame
        webcam_out.write(frame)

        if is_first_frame:
            print("Webcam RECORDING STARTED")
            log_file.write(f'WEBCAM_REC_STARTED {int(time.time() * 1000)}\n')
            is_first_frame = False

        # capture screen
        screen_frame = pyautogui.screenshot()
        screen_frame = np.array(screen_frame)
        screen_frame = cv2.cvtColor(screen_frame, cv2.COLOR_RGB2BGR)
        screen_out.write(screen_frame)
        
        # Display webcam frame
        cv2.imshow('Webcam', frame)
        
        if cv2.waitKey(1) == ord('s'):
            log_file.write(f'WEBCAM_REC_ENDED {int(time.time() * 1000)}\n')
            print("Webcam RECORDING ENDED")
            break

    capture.release()
    webcam_out.release()
    screen_out.release()
    cv2.destroyAllWindows()
    log_file.close()

    print(f"The recorded webcam video file is available from here: {webcam_video_file_path}")
    print(f"The recorded screen video file is available from here: {screen_video_file_path}")
    print(f"The corresponding log file is available from here: {log_file_path}")


if __name__ == "__main__":
    filename = args.fid
    if args.filename_prefix != "":
        filename = f"{args.filename_prefix}_{filename}"
    if args.filename_postfix != "":
        filename = f"{filename}_{args.filename_postfix}"
    record_video(filename=filename,
                 webcam_number=args.webcam_number, fps=args.fps, dir=args.dir)
