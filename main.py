import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QDial,
    QDoubleSpinBox,
    QFormLayout,
    QLineEdit,
    QMessageBox,
    QHBoxLayout
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class DiabetesPredictionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Style settings
        self.setStyleSheet("""
                            QMainWindow {
                                background-color: #108F88;
                            }
                            QPushButton {
                                background-color: #8150A4;
                                color: white;
                                padding: 10px;
                            }
                            QPushButton:hover {
                                background-color: #45a049;
                            }
                            QLineEdit, QSpinBox, QDoubleSpinBox, QSlider, QDial {
                                padding: 5px;
                            }
                            QLabel {
                                font-weight: bold;
                            }
                        """)

        # Window settings
        self.setWindowTitle("Diabetes Prediction App")
        self.setGeometry(100, 100, 400, 600)
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Menu for data import
        self.menu_bar = self.menuBar()
        self.file_menu = self.menu_bar.addMenu("&File")
        self.import_action = QAction("Import Data", self)
        self.import_action.triggered.connect(self.import_data)
        self.file_menu.addAction(self.import_action)

        # Load data
        self.data = None
        self.load_data_at_startup()

        # Input fields
        self.feature_inputs = {}
        self.feature_labels = {}
        self.input_layout = QFormLayout()
        self.create_input_fields()

        # Prediction button
        self.predict_button = QPushButton("Predict Diabetes")
        self.predict_button.clicked.connect(self.predict_diabetes)
        self.layout.addWidget(self.predict_button)

        # Result display
        self.result_label = QLabel("Prediction Result:")
        self.result_display = QLineEdit()
        self.result_display.setReadOnly(True)
        self.layout.addWidget(self.result_label)
        self.layout.addWidget(self.result_display)

        # Plotting setup
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.hide()
        self.layout.addWidget(self.canvas)

        # Model training
        self.clf = None
        if self.data is not None:
            self.clf = self.train_model()

    def load_data_at_startup(self):
        try:
            self.data = pd.read_csv('diabetes.csv')
            self.analyze_data()
        except Exception as e:
            QMessageBox.warning(self, "Loading Error", f"Failed to load data at startup: {e}")

    def import_data(self):
        try:
            self.data = pd.read_csv('diabetes.csv')  # Replace with a file dialog if needed
            self.analyze_data()
            self.clf = self.train_model()
        except Exception as e:
            QMessageBox.warning(self, "Import Error", f"Failed to import data: {e}")

    def analyze_data(self):
        if self.data is not None:
            info_str = self.data.info(buf=None)
            desc_str = str(self.data.describe())
            corr_str = str(self.data.corr())

            self.show_data_analysis(info_str, desc_str, corr_str)
            self.visualize_data()
        else:
            QMessageBox.warning(self, "Data Error", "No data available for analysis.")

    def show_data_analysis(self, info_str, desc_str, corr_str):
        QMessageBox.information(self, "Data Analysis",
                                f"Info:\n{info_str}\n\nDescribe:\n{desc_str}\n\nCorrelation:\n{corr_str}")

    def visualize_data(self):
        self.visualization_window = QMainWindow(self)
        self.visualization_window.setWindowTitle("Data Visualization")
        central_widget = QWidget(self.visualization_window)
        self.visualization_window.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        hist_canvas = FigureCanvas(Figure(figsize=(5, 4)))
        self.data.hist(ax=hist_canvas.figure.subplots(), bins=5)
        hist_canvas.figure.tight_layout()
        layout.addWidget(hist_canvas)

        corr_canvas = FigureCanvas(Figure(figsize=(5, 4)))
        sns.heatmap(self.data.corr(), annot=True, fmt=".2f", ax=corr_canvas.figure.subplots())
        layout.addWidget(corr_canvas)

        self.visualization_window.show()

    def create_input_fields(self):
        for feature in ["Pregnancies", "Glucose", "Blood_Pressure", "Skin_Thickness", "Insulin", "BMI",
                        "Diabetes_Pedigree_Function", "Age"]:
            label = QLabel(f"{feature}:")
            input_widget = self.create_input_widget(feature)

            if isinstance(input_widget, (QSlider, QDial)):
                value_label = QLabel("0")
                self.feature_labels[feature] = value_label
                row_layout = QHBoxLayout()
                row_layout.addWidget(input_widget)
                row_layout.addWidget(value_label)
                self.input_layout.addRow(label, row_layout)
            else:
                self.input_layout.addRow(label, input_widget)

            self.feature_inputs[feature] = input_widget

        self.layout.addLayout(self.input_layout)

    def create_input_widget(self, feature):
        if feature == "Glucose":
            input_widget = QSpinBox()
            input_widget.setMinimum(60)
            input_widget.setMaximum(200)
        elif feature == "Blood_Pressure":
            input_widget = QSpinBox()
            input_widget.setMinimum(0)
            input_widget.setMaximum(150)
        elif feature == "Insulin":
            input_widget = QSpinBox()
            input_widget.setMinimum(0)
            input_widget.setMaximum(999)
        elif feature == "BMI":
            input_widget = QDoubleSpinBox()
            input_widget.setMaximum(60.0)
            input_widget.setDecimals(2)
        elif feature == "Diabetes_Pedigree_Function":
            input_widget = QDoubleSpinBox()
            input_widget.setMaximum(3.0)
            input_widget.setDecimals(3)
        elif feature == "Skin_Thickness":
            input_widget = QDial()
            input_widget.setMinimum(10)
            input_widget.setMaximum(50)
            input_widget.valueChanged.connect(lambda value, f=feature: self.update_display_label(f, value))
        elif feature == "Pregnancies":
            input_widget = QSlider(Qt.Orientation.Horizontal)
            input_widget.setMinimum(0)
            input_widget.setMaximum(20)
            input_widget.valueChanged.connect(lambda value, f=feature: self.update_display_label(f, value))
        elif feature == "Age":
            input_widget = QSlider(Qt.Orientation.Horizontal)
            input_widget.setMinimum(10)
            input_widget.setMaximum(99)
            input_widget.valueChanged.connect(lambda value, f=feature: self.update_display_label(f, value))
        else:
            input_widget = QLineEdit()
        return input_widget

    def update_display_label(self, feature, value):
        if feature in self.feature_labels:
            self.feature_labels[feature].setText(str(value))

    def train_model(self):
        x_train, x_test, y_train, y_test = train_test_split(
            self.data[["Pregnancies", "Glucose", "Blood_Pressure", "Skin_Thickness", "Insulin", "BMI",
                       "Diabetes_Pedigree_Function", "Age"]],
            self.data["Outcome"],
            test_size=0.2,
            random_state=42
        )
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(x_train, y_train)
        return clf

    def predict_diabetes(self):
        if self.clf is None:
            QMessageBox.warning(self, "Model Error", "The prediction model is not trained.")
            return

        feature_values = [self.get_input_value(feature) for feature in
                          ["Pregnancies", "Glucose", "Blood_Pressure", "Skin_Thickness", "Insulin", "BMI",
                           "Diabetes_Pedigree_Function", "Age"]]
        x = pd.DataFrame([feature_values],
                         columns=["Pregnancies", "Glucose", "Blood_Pressure", "Skin_Thickness", "Insulin", "BMI",
                                  "Diabetes_Pedigree_Function", "Age"])
        probabilities = self.clf.predict_proba(x)
        result = f"Probability of being Diabetic: {probabilities[0][1]:.2%}"
        self.result_display.setText(result)
        self.canvas.show()
        self.plot_probabilities_pie(probabilities)

    def get_input_value(self, feature):
        widget = self.feature_inputs[feature]
        if isinstance(widget, (QSpinBox, QDoubleSpinBox, QSlider, QDial)):
            return widget.value()
        elif isinstance(widget, QLineEdit):
            return float(widget.text())
        else:
            return 0

    def plot_probabilities_pie(self, probabilities):
        labels = ["Not Diabetic", "Diabetic"]
        colors = ['#ff9999', '#66b3ff']
        explode = (0.1, 0)
        plt.cla()
        self.ax.pie(probabilities[0], labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, explode=explode)
        self.ax.axis('equal')
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = DiabetesPredictionApp()
    mainWin.show()
    sys.exit(app.exec())
