def build_sessions(df, user_col='uuid', time_col='timestamp', max_gap=30*60*1000, max_session_len=20):
    sessions = []
    df = df.sort_values([user_col, time_col])
    for uuid, group in df.groupby(user_col):
        group = group.sort_values(time_col)
        session = []
        prev_time = None
        for idx, row in group.iterrows():
            if prev_time is None or (row[time_col] - prev_time > max_gap) or (len(session) >= max_session_len):
                if session:
                    sessions.append(session)
                session = []
            session.append(row)
            prev_time = row[time_col]
        if session:
            sessions.append(session)
    return sessions
