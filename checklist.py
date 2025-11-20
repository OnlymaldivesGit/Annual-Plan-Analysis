def regulatory(rolling_df):
    issue_1=rolling_df[rolling_df["Working days"]>4].reset_index(drop=True)
    issue=rolling_df[(rolling_df["Working days"]<=4) & (rolling_df["Working days"]>2)].reset_index(drop=True)
    issue["Consecutive working"] = issue.apply(lambda row: count_consecutive(row["Values"], ["1"]), axis=1)
    issue=issue[(issue["Consecutive Non working"]<3) & (issue["Consecutive working"]<4) ].reset_index(drop=True)
    issue=issue[~((issue["Working days"] == 3) & (issue["Consecutive working"] == 3))]

    violation_df=pd.concat([issue, issue_1], ignore_index=True, join='inner')
    return violation_df

