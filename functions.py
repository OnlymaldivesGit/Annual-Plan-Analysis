import pandas as pd
from datetime import date, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

Management_crew=['HJIF', 'MKHA', 'ISMS','AUZU', 'MRAS','ARMO', 'AHNA', 'DWOO']
trainings=[ '-', 'GRND 300', 'CRM', 'GRND 400', 'UPRT', 'MEDICAL', 'FLT TRNG', 'ESET', 'FMT', 'LPC', 'LINE CHECK', 'DGR', 'SMS', 'EVAC', 'FIRE', 'LC -400', 'OPC', 'RSQT', 'RSQC', 'LICENCE', 'TRE CERT', 'TRI Rating', 'TREMTR', 'TRIMTR', ' FLT TRNG', '', '   ', 'ELP']


cols = ["Crew code","Crew Name"]

start_date = date(2026, 1, 1)
end_date = date(2026, 12, 31)
delta = timedelta(days=1)

current_date = start_date
while current_date <= end_date:
    cols.append(current_date)
    current_date += delta


rows=["Optimized","Schedulers","Master","Optimized 2026","Master's 2026","Scheduler's 2025","Optimized 2026 Without Trainings","14 Pattern\n2026","7+7 Pattern\n2026"]


def safe_date_format(col_name):
    try:
        return pd.to_datetime(col_name).strftime('%Y-%m-%d')
    except (ValueError, TypeError):
        return col_name


def tma_df_processing(df):
    df.columns = [safe_date_format(col) for col in df.columns]
    df = df.loc[:, ~df.columns.str.startswith('Unnamed:')]
    df=df[df["Type"]=="Captain"]
    df.drop(["Type"],axis=1,inplace=True)
    df=df[~df["Crew code"].isin(Management_crew)]

    id_columns = ['Crew code', 'Crew Name']
    date_columns = [col for col in df.columns if col not in id_columns]

    tma_data = pd.melt(df,id_vars=id_columns,value_vars=date_columns,var_name='Date',value_name='Status')

    tma_data=tma_data[(tma_data["Date"]>"2025-12-31") & (tma_data["Date"]<"2027-01-01")].reset_index(drop=True)
    tma_data['Date'] = pd.to_datetime(tma_data['Date'])
    tma_data['Status'] = tma_data['Status'].replace("PAL", "AL")
    return tma_data

def schedule_processing(df):
    df.columns=cols
    df = df.dropna(subset=['Crew code'])
    df=df[~df["Crew code"].isin(rows)]
    return df

def melt_fun(df):
  id_columns = ['Crew code', 'Crew Name']
  date_columns = [col for col in df.columns if col not in id_columns]

  df_long = pd.melt(
      df,
      id_vars=id_columns,
      value_vars=date_columns,
      var_name='Date',
      value_name='Status')

  df_long['Status'] = df_long['Status'].replace("1", 1)
  df_long['Status'] = df_long['Status'].fillna("-")
  df_long['Date'] = pd.to_datetime(df_long['Date'])

  return df_long


def count_consecutive(values,code):
    """
    Count the maximum number of consecutive 'X' values in an array.
    """
    if len(values) == 0:
        return 0

    max_consecutive = 0
    current_consecutive = 0

    for value in values:
        if value in code:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0

    return max_consecutive


def count_consecutive_1(values):
    """
    Count the maximum number of consecutive 'X' values in an array.
    """
    if len(values) == 0:
        return 0

    max_consecutive = 0
    current_consecutive = 0

    for value in values:
        if value == '1':
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0

    return max_consecutive

def add_rolling_counts(df, window_days=7):

    result_df = df.copy()
    result_df['Date'] = pd.to_datetime(result_df['Date'])

    # Sort by crew and date
    result_df = result_df.sort_values(['Crew code', 'Date']).reset_index(drop=True)

    # Status columns to analyze
    status_columns = 'Status'
    result_df = result_df[["Crew code", "Crew Name", "Date", status_columns]]

    # Initialize new columns
    result_df['start_date'] = pd.NaT
    result_df['end_date'] = pd.NaT
    result_df['Values'] = None

    for idx, row in result_df.iterrows():
        crew_code = row['Crew code']
        start_date = row['Date']
        end_date = start_date + timedelta(days=window_days - 1)

        mask = result_df[
              (result_df['Crew code'] == crew_code) &
              (result_df['Date'] >= start_date) &
              (result_df['Date'] <= end_date)
          ]

        result_df.at[idx, 'start_date'] = start_date
        result_df.at[idx, 'end_date'] = end_date
        result_df.at[idx, 'Values'] = mask[status_columns].astype(str).tolist()

    result_df["OFF days"] = result_df["Values"].apply(lambda x: x.count("X"))
    result_df["Working days"] = result_df["Values"].apply(lambda x: x.count("1"))
    result_df["Annual Leaves days"] = result_df["Values"].apply(lambda x: x.count("AL"))
    result_df["Other days"] = result_df["Values"].apply(lambda x: x.count("-"))
    result_df["Consecutive Non working"] = result_df.apply(lambda row: count_consecutive(row["Values"], ["X","AL","-"]), axis=1)
    result_df["Total"]=result_df["OFF days"]+result_df["Working days"]+result_df["Annual Leaves days"]+result_df["Other days"]
    result_df=result_df[result_df["end_date"]<"2027-01-01"]

    return result_df


def vacation_fun(optimzed_df):
  vacation_optimized = optimzed_df[optimzed_df["Status"].isin(["AL","PAL","AU"])] \
    .sort_values(by=["Crew code", "Crew Name",  "Date","Status"]) \
    .reset_index(drop=True)

  vacation_optimized['Date'] = pd.to_datetime(vacation_optimized['Date'])

  # # Compute diff within each group
  vacation_optimized['diff'] = vacation_optimized.groupby(['Crew code', "Crew Name", "Status"])['Date'].diff().dt.days
  vacation_optimized['group'] = (vacation_optimized['diff'] != 1).groupby([vacation_optimized['Crew code'], vacation_optimized['Crew Name']]).cumsum()

  # Aggregate only within correct vacation groups
  vacation_optimized = vacation_optimized.groupby(['Crew code', "Crew Name", 'group', "Status"]).agg(
      start_date=('Date', 'min'),
      end_date=('Date', 'max'),
      Duration=('Date', 'count')
  ).reset_index()


  vacation_optimized = vacation_optimized.sort_values(['Crew code', 'group'])


  vacation_optimized['next_start_date'] = vacation_optimized.groupby('Crew code')['start_date'].shift(-1)
  vacation_optimized['Days gap with next vacation'] = (pd.to_datetime(vacation_optimized['next_start_date']) - pd.to_datetime(vacation_optimized['end_date'])).dt.days -1
  vacation_optimized.drop(['next_start_date'],axis=1,inplace=True)

  return vacation_optimized

  






