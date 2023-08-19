import tkinter as tk

class Window:
  def __init__(self):
    self.frame = tk.Tk()
    self.frame.title("Metronome")
    self.frame.geometry("400x150")
  
  def set_slider(self, slider_function):
    self.slider = tk.Scale(self.frame, from_=40, to=200, orient='horizontal', command=slider_function)
    self.slider.set(120)
    self.slider.pack(fill='x', padx=20, pady=20)
  
  def set_start_button(self, start_function):
    self.start_btn = tk.Button(self.frame, text="Start", command=start_function)
    self.start_btn.pack(side=tk.LEFT, padx=20, pady=0)
  
  def set_stop_button(self, stop_function):
    self.stop_btn = tk.Button(self.frame, text="Stop", command=stop_function)
    self.stop_btn.pack(side=tk.RIGHT, padx=20, pady=0)
  
  def show(self):
    self.frame.mainloop()