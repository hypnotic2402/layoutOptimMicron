from tkinter import *
import datetime
import tkinter as tk
from tkinter import filedialog
from tkVideoPlayer import TkinterVideo

root = tk.Tk()
root.title = ' Placement Process'

root.geometry("640x480")
root.configure(bg = "#FFFFFF")

frame = tk.Frame(root)
frame.pack()

top_frame = tk.Frame(root, bg = "#68C6FB")
top_frame.pack(fill = "both", side = "top")

Text_Title = tk.Label(top_frame, text = "Placement Process", bg = "#68C6FB", font = ("Helvatica", 15, "bold"))
Text_Title.pack(side = "left")

vid_player = TkinterVideo(root, scaled = True)
vid_player.pack(expand = True, fill = "both")

bottom_frame = tk.Frame(root, bg="#FFFFFF")
bottom_frame.pack(fill="both", side = "bottom")

def update_duration(event):
    duration = vid_player.video_info()["duration"]
    end_time["text"] = str(datetime.timedelta(seconds=duration))
    progress_slider["to"] = duration

def update_scale(event):
    progress_value.set(vid_player.current_duration())

def load_video():
    file_path = filedialog.askopenfilename()
    if file_path:
        vid_player.load(file_path)
        progress_slider.config(to = 0, from_= 0)
        Play_Pause_Btn["text"] = "Play"
        progress_value.set(0)

def seek(value):
    vid_player.seek(int(value))

def play_pause_func():
    if vid_player.is_paused():
        vid_player.play()
        Play_Pause_Btn["text"] = "Pause"

    else:
        vid_player.pause()
        Play_Pause_Btn["text"] = "Play"

def video_ended(event):
    progress_slider.set(progress_slider["to"])
    Play_Pause_Btn["text"] = "Play"
    progress_slider.set(0)


start_time = tk.Label(root, text=str(datetime.timedelta(seconds=0)))
start_time.pack(side = "left")

progress_value = tk.IntVar(root)
progress_slider = tk.Scale(root, variable = progress_value, from_=0, to = 0, orient = "horizontal", command= seek)
progress_slider.pack(side = "left", fill = "x", expand = True)

end_time = tk.Label(root, text=str(datetime.timedelta(seconds=0)))
end_time.pack(side = "left")

# Play_Button = PhotoImage(file = "button_1.png")
Play_Pause_Btn = tk.Button(bottom_frame, text = "Play", bg = "#FFFFFF", font = ("Calibri", 12, "bold"), command = play_pause_func).pack(ipadx = 10, side="left")

# Browse = PhotoImage(file = "button_3.png")
Browse_btn = tk.Button(bottom_frame, text = "Browse", bg = "#FFFFFF", font = ("Calibri", 12, "bold"), command = load_video).pack(padx = 5, ipadx = 10, side="left")

macro_count = 200

Macro_Place = tk.Label(bottom_frame, text = "No of Macros: " + str(macro_count), font = ("Calibri", 12, "bold")).pack(ipadx = 10, side="right")

vid_player.bind("<<Duration>>", update_duration)
vid_player.bind("<<SecondChanged>>", update_scale)
vid_player.bind("<<Ended>>", video_ended)

root.resizable(False, False)
root.mainloop();
