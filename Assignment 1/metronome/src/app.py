from src.metronome import Metronome
from src.window import Window

class App:
  def __init__(self):
    self.metronome = Metronome()
    self.window = Window()
    
    self.window.set_slider(self.metronome.change_tempo)
    self.window.set_start_button(self.metronome.start)
    self.window.set_stop_button(self.metronome.stop)
  
  def start(self):
    self.window.show()