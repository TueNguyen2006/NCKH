import time
import threading
import serial
import RPi.GPIO as GPIO

class Ras:
    def __init__(self, serial_port="/dev/serial0", baudrate=9600, vibration_motor=17):
        """Initialize Raspberry Pi hardware for audio and vibration alert."""
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.ser = None
        self.serial_ready = threading.Event()
        self.vibration_motor = vibration_motor
        self.running_event = threading.Event()  # Controls the alert (audio + vibration)

        # Setup GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.vibration_motor, GPIO.OUT)

        # Start serial connection in a separate thread
        self.serial_thread = threading.Thread(target=self._serial_connect, daemon=True)
        self.serial_thread.start()

        # Start threads for music and motor
        self.music_thread = threading.Thread(target=self._play_music, daemon=True)
        self.motor_thread = threading.Thread(target=self._vibrate_motor, daemon=True)
        self.music_thread.start()
        self.motor_thread.start()

    def _serial_connect(self):
        """Continuously try to connect to Serial until successful."""
        while True:
            try:
                self.ser = serial.Serial(port=self.serial_port, baudrate=self.baudrate, timeout=1)
                print("Serial connected successfully!")
                self.serial_ready.set()
                break
            except serial.SerialException as e:
                print("Serial connection error: %s", e)
                time.sleep(1)

    def _get_serial(self):
        """Wait until serial is ready and return it."""
        self.serial_ready.wait()
        return self.ser

    def _send_command(self, command):
        """Send command bytes via serial."""
        ser = self._get_serial()
        ser.write(command)

    def play_song(self, song_number):
        """Send command to play song with given song number."""
        command = bytes([0x7E, 0xFF, 0x06, 0x03, 0x00, 0x00, song_number, 0xEF])
        self._send_command(command)

    def set_volume(self, volume_level):
        """Send command to set volume (0-30)."""
        command = bytes([0x7E, 0xFF, 0x06, 0x06, 0x00, 0x00, volume_level, 0xEF])
        self._send_command(command)

    def _play_music(self, volume_level=30, song_number=1):
        """Continuously play music when alert is activated."""
        while True:
            self.running_event.wait()
            print("Playing alert music...")
            self.set_volume(volume_level)
            self.play_song(song_number)
            time.sleep(2)  # Delay between songs

    def _vibrate_motor(self, duration=1):
        """Continuously vibrate motor when alert is activated."""
        while True:
            self.running_event.wait()
            print("Activating vibration motor...")
            GPIO.output(self.vibration_motor, GPIO.HIGH)
            time.sleep(duration)
            GPIO.output(self.vibration_motor, GPIO.LOW)
            time.sleep(0.5)

    def warning(self):
        """Activate alert if not already running."""
        if self.running_event.is_set():
            print("Alert already active.")
            return
        print("Activating alert!")
        self.running_event.set()

    def turn_off(self):
        """Deactivate alert if active."""
        if not self.running_event.is_set():
            print("Alert is not active; nothing to turn off.")
            return
        print("Deactivating alert...")
        self.running_event.clear()

    def close(self):
        """Clean up hardware resources."""
        self.turn_off()
        if self.ser:
            self.ser.close()
        GPIO.cleanup()
        print("Hardware resources cleaned up.")