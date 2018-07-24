import BSDE
import warnings
import bokeh
from bokeh.layouts import gridplot
from bokeh.plotting import figure, output_notebook, show, output_file
import time
from statics import State
import config
import pandas as pd


def main():
    args = config.ArgParser.from_cmd()
    state = State.get_from_args(args)
    regression_params = RegressionParams.from_state()
    options_df = pd.DataFrame.from_csv('options.csv')
    stocks = [Option.build_from_csv_row(row) for row in options_df]