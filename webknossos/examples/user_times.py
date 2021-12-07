import pandas as pd

import webknossos as wk


def main() -> None:
    df = pd.DataFrame()
    df.columns = pd.MultiIndex([[], []], [[], []], names=("year", "month"))
    df.index.name = "email"

    users = wk.User.get_all_managed_users()
    for user in users:
        for logged_time in user.get_logged_times():
            df.loc[
                user.email, (logged_time.year, logged_time.month)
            ] = logged_time.duration_in_seconds

    df = df.fillna(0).astype("uint")
    df = df.sort_index(axis="index").sort_index(axis="columns")

    year = 2021
    has_logged_times_in_year = df.loc[:, year].sum(axis="columns") != 0

    print(f"Logged User Times {year}:\n")
    print(df.loc[has_logged_times_in_year, year].to_markdown())


if __name__ == "__main__":
    main()
