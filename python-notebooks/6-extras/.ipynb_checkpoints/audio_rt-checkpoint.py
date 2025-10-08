import numpy as np
import sounddevice as sd
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
import threading

# Parámetros de audio
fs = 44100/8  # Frecuencia de muestreo
duration = 0.1  # Duración de cada captura en segundos
N = int(fs * duration)  # Número de muestras por captura

# Configuración de la ventana
app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(title="Espectro de Audio en Tiempo Real")
win.resize(800, 600)

# Dominio del tiempo
time_plot = win.addPlot(title="Señal en el dominio del tiempo")
time_curve = time_plot.plot(pen='y')
time_plot.setLabel('left', 'Amplitud')
time_plot.setLabel('bottom', 'Tiempo', units='s')
time_plot.setXRange(0, duration)
time_plot.setYRange(-1, 1)

# Dominio de la frecuencia
win.nextRow()
freq_plot = win.addPlot(title="Espectro de frecuencia")
freq_curve = freq_plot.plot(pen='c')
freq_plot.setLabel('left', 'Magnitud')
freq_plot.setLabel('bottom', 'Frecuencia', units='Hz')
freq_plot.setXRange(0, fs / 2)
freq_plot.setYRange(0, 0.5)

# Variables globales para la señal
time = np.linspace(0, duration, N, endpoint=False)
freq = np.fft.rfftfreq(N, 1 / fs)
audio_data = np.zeros((N, 1))  # Datos iniciales en silencio

audio_lock = threading.Lock()
update_counter = 0

def audio_callback(indata, frames, time, status):
    global audio_data
    if status:
        print(f"Stream status: {status}")
    with audio_lock:
        audio_data = indata.copy()

class GraphUpdateThread(QtCore.QThread):
    def run(self):
        global update_counter
        while True:
            with audio_lock:
                current_data = audio_data.copy()

            if update_counter % 2 == 0:  # Actualizar cada 2 ciclos
                time_curve.setData(time, current_data[:, 0])
                spectrum = np.abs(np.fft.rfft(current_data[:, 0]) / N)
                freq_curve.setData(freq, spectrum)

            update_counter += 1
            QtCore.QThread.msleep(10)


# Crear el hilo de actualización
graph_thread = GraphUpdateThread()

# Captura de audio en tiempo real
stream = sd.InputStream(
    channels=1,
    samplerate=fs,
    blocksize=N,
    callback=audio_callback
)

try:
    with stream:
        # Inicia el hilo de gráficos
        graph_thread.start()
        # Ejecuta la ventana de PyQtGraph
        win.show()
        app.exec_()
except KeyboardInterrupt:   
    print("\nAplicación cerrada.")
finally:
    graph_thread.terminate()
