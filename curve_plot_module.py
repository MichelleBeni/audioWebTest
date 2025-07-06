
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def create_curve_plot(
    df,
    x_col,
    y_col,
    new_x,
    xlabel,
    ylabel,
    filename,
    save_dir="static/plots"
):
    os.makedirs(save_dir, exist_ok=True)

    # חישוב ממוצעים
    grouped = df.groupby(y_col)[x_col].mean().reset_index()
    x = grouped[x_col].values
    y = grouped[y_col].values

    # התאמת פולינום מדרגה 2
    coeffs = np.polyfit(x, y, 2)
    poly = np.poly1d(coeffs)

    # ערכים לגרף
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = poly(x_fit)

    # חישוב ציון גם אם מחוץ לטווח, עם קליפינג ל-0–5
    predicted_y = poly(new_x)
    predicted_y = max(0, min(5.0, predicted_y))  # גבולות 0–5.0

    # גרף
    plt.figure()
    plt.plot(x_fit, y_fit, color='blue')  # קו מגמה
    plt.scatter(x, y, color='black')      # נקודות ממוצע
    plt.scatter(new_x, predicted_y, color='red')  # נקודה של ההקלטה
    plt.axvline(x=new_x, color='red', linestyle='dashed')
    plt.axhline(y=predicted_y, color='red', linestyle='dashed')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} vs. {xlabel}')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight')
    plt.close()
