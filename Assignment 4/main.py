from argparse import ArgumentParser
import librosa
import numpy as np
import pandas as pd
import pickle as pkl
import warnings

STEP_SIZE = 0.02
SKIP_SIZE = 6


class Parser:
    def __init__(self):
        self.__parser = ArgumentParser(description="Code for Assignment 4")
        self.__add_args()
        self.__parse_args()
        self.__validate_args()
        return

    def __add_args(self):
        self.__parser.add_argument(
            "-i", "--input", dest="input", help="Input wav file", metavar="FILE"
        )
        self.__parser.add_argument(
            "-o", "--output", dest="output", help="Output csv file", metavar="FILE"
        )
        return

    def __parse_args(self):
        args = self.__parser.parse_args()
        self.__input_file: str = args.input
        self.__output_file: str = args.output
        return

    def __validate_args(self):
        if not self.__input_file:
            self.__print_error("Input file is required")
        if not self.__input_file.endswith(".wav"):
            self.__print_error('Input file must end with ".wav"')
        if not self.__output_file:
            self.__print_error("Output file is required")
        if not self.__output_file.endswith(".csv"):
            self.__print_error('Output file must end with ".csv"')
        return

    @staticmethod
    def __print_error(error):
        print(f"\033[91m[ERROR]  :\033[0m {error}")
        print(f"\033[96m[INFO]   :\033[0m Use -h or --help for help")
        exit(1)

    def get_args(self):
        print(f"\033[92m[SUCCESS]:\033[0m Parsed arguments")
        return self.__input_file, self.__output_file


class Predictor:
    def __init__(self, model_file):
        self.__load_model(model_file)
        return

    def __load_model(self, model_file):
        try:
            with open(model_file, "rb") as file:
                self.__model = pkl.load(file)
                print(f"\033[92m[SUCCESS]:\033[0m Model loaded")
        except FileNotFoundError:
            print(f"\033[91m[ERROR]  :\033[0m Model file not found")
            print("           Please run train.ipynb first")
            exit(1)
        finally:
            return

    def __get_duration(self):
        y, sr = librosa.load(self.__input_file)
        duration = librosa.get_duration(y=y, sr=sr)
        return duration

    def __get_mfccs(self, start_time):
        y, sr = librosa.load(self.__input_file, offset=start_time, duration=0.01)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.array(mfccs.T)

    def __make_prediction(self, start_time):
        try:
            x_data = self.__get_mfccs(start_time)
            y_data = self.__model.predict(x_data)
            return y_data[0]
        except Exception as e:
            print(f"\033[91m[ERROR]  :\033[0m {e}")
            exit(1)

    def __post_processing(self, data):
        for i in range(len(data) - SKIP_SIZE):
            for j in range(1, SKIP_SIZE):
                if data[i][2] == 1 and data[i + j][2] == 1:
                    for k in range(1, j):
                        data[i + k][2] = 1
        final_data = []
        start = data[0][0]
        pred_value = data[0][2]
        for d in data[1:]:
            current_pred = d[2]
            if current_pred == pred_value:
                continue
            final_data.append([start, d[0], True if pred_value == 1 else False])
            start = d[0]
            pred_value = current_pred
        final_data.append([start, data[-1][1], True if pred_value == 1 else False])
        return final_data

    def __save_predictions(self, final_data):
        df = pd.DataFrame(final_data)
        df.columns = ["Start Time", "End Time", "Silence"]
        df.to_csv(self.__output_file, index=False)
        print(f"\033[92m[SUCCESS]:\033[0m Predictions made")
        return """  """

    def predict(self, input_file, output_file):
        self.__input_file = input_file
        self.__output_file = output_file
        duration = self.__get_duration()
        pred_data = []
        i = 0
        current_pred = 0
        while i + STEP_SIZE <= duration:
            current_pred = self.__make_prediction(i)
            pred_data.append([round(i, 3), round(i + STEP_SIZE, 3), current_pred])
            i += STEP_SIZE
        pred_data.append([pred_data[-1][1], round(duration, 3), current_pred])
        final_data = self.__post_processing(pred_data)
        self.__save_predictions(final_data)
        return


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = Parser()
    input_file, output_file = parser.get_args()
    predictor = Predictor("model.pkl")
    predictor.predict(input_file, output_file)
