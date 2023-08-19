import time
import threading

class Metronome:
  def __init__(self):
    self.tempo = 120
    self.interval = 60 / self.tempo
    self.running = False
    self.thread = None
  
  def change_tempo(self, new_tempo):
    self.tempo = int(new_tempo)
    self.interval = 60 / self.tempo
  
  def beat(self):
    if self.running:
      print('[METRONOME] Tick')
      time.sleep(self.interval)
      self.beat()
  
  def start(self):
    if not self.running:
      print('[METRONOME] Start')
      self.running = True
      self.thread = threading.Thread(target=self.beat)
      self.thread.start()
  
  def stop(self):
    if self.running:
      print('[METRONOME] Stop')
      self.running = False
      self.thread.join()
      self.thread = None