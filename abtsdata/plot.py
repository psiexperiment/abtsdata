import pandas as pd


def add_weekends(ax, date_start, date_end):
    for date in pd.date_range(date_start, date_end):
        if date.day_of_week == 5:
            ax.axvspan(date - pd.Timedelta(days=0.5), date + pd.Timedelta(days=1.5), alpha=0.5)
            ax.text(date + pd.Timedelta(days=0.5), 0.75, 'weekend', transform=ax.get_xaxis_transform(), rotation=90, ha='center', va='center')
