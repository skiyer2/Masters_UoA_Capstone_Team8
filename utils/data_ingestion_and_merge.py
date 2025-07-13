def ingestion():
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    import os

    data_dir = os.path.join(os.getcwd(), "data")
    output_dir = os.path.join(os.getcwd(), "outbrain")
    os.makedirs(output_dir, exist_ok=True)

    clicks_train = pd.read_csv(os.path.join(data_dir, 'clicks_train.csv')).iloc[:5000000, :]
    documents_categories = pd.read_csv(os.path.join(data_dir, 'documents_categories.csv'))
    documents_entities = pd.read_csv(os.path.join(data_dir, 'documents_entities.csv'))
    documents_meta = pd.read_csv(os.path.join(data_dir, 'documents_meta.csv'))
    events = pd.read_csv(os.path.join(data_dir, 'events.csv'))
    page_views_sample = pd.read_csv(os.path.join(data_dir, 'page_views_sample.csv'))
    promoted_content = pd.read_csv(os.path.join(data_dir, 'promoted_content.csv'))



    unique_ad_ids = clicks_train['ad_id'].unique().tolist()
    promoted_content = promoted_content[promoted_content['ad_id'].isin(unique_ad_ids)]
    print("Step 1: Merging clicks_train with promoted_content...")
    df = clicks_train.merge(promoted_content, how='left', on='ad_id')
    df.rename(columns={'document_id': 'ad_document_id'}, inplace=True)

    unique_display_ids = events['display_id'].unique().tolist()
    events = events[events['display_id'].isin(unique_display_ids)]
    print("Step 2: Merging with events...")
    df = df.merge(events, how='left', on='display_id')
    df.rename(columns={'document_id': 'event_document_id', 'timestamp': 'event_timestamp'}, inplace=True)

    unique_ad_document_ids = df['ad_document_id'].unique().tolist()
    documents_meta = documents_meta[documents_meta['document_id'].isin(unique_ad_document_ids)]
    print("Step 3a: Merging ad_document_id with documents_meta...")
    df = df.merge(documents_meta, how='left', left_on='ad_document_id', right_on='document_id')
    df.rename(columns=lambda x: f"{x}_admeta" if x in ['source_id', 'published_id', 'publish_time'] else x, inplace=True)
    df.drop(columns=['document_id'], inplace=True)

    unique_event_document_id = df['event_document_id'].unique().tolist()
    documents_meta  = documents_meta[documents_meta['document_id'].isin(unique_event_document_id)]
    print("Step 3b: Merging event_document_id with documents_meta...")
    df = df.merge(documents_meta, how='left', left_on='event_document_id', right_on='document_id')
    df.rename(columns=lambda x: f"{x}_eventmeta" if x in ['source_id', 'published_id', 'publish_time'] else x, inplace=True)
    df.drop(columns=['document_id'], inplace=True)

    unique_ad_document_id = df['ad_document_id'].unique().tolist()
    documents_categories  = documents_categories[documents_categories['document_id'].isin(unique_ad_document_id)]
    print("Step 4a: Merging with documents_categories...")
    df = df.merge(documents_categories, how='left', left_on='ad_document_id', right_on='document_id', suffixes=('', '_cat'))

    unique_ad_document_id = df['ad_document_id'].unique().tolist()
    documents_entities  = documents_entities[documents_entities['document_id'].isin(unique_ad_document_id)]
    print("Step 4b: Merging with documents_entities...")
    df = df.merge(documents_entities, how='left', left_on='ad_document_id', right_on='document_id', suffixes=('', '_ent'))

    unique_uuid = df['uuid'].unique().tolist()
    unique_event_document_id= df['event_document_id'].unique().tolist()
    page_views_sample  = page_views_sample[page_views_sample['uuid'].isin(unique_uuid)]
    page_views_sample  = page_views_sample[page_views_sample['document_id'].isin(unique_event_document_id)]
    print("Step 5: Merging with page_views_sample...")
    df = df.merge(page_views_sample, how='left', left_on=['uuid', 'event_document_id'], right_on=['uuid', 'document_id'])

    print("Step 6: Extracting temporal features...")
    df['event_timestamp'] = pd.to_datetime(df['event_timestamp'], unit='ms')
    df['hour'] = df['event_timestamp'].dt.hour
    df['dayofweek'] = df['event_timestamp'].dt.dayofweek
    df = df.sort_values(by=['uuid', 'event_timestamp']).reset_index(drop=True)
    df['time_diff'] = df.groupby('uuid')['event_timestamp'].diff().dt.total_seconds().div(60)
    df['new_session'] = (df['time_diff'] > 30) | (df['time_diff'].isna())
    df['session_id'] = df.groupby('uuid')['new_session'].cumsum()

    df = df.iloc[:5000000,:]

    df_clicked = df[df['clicked'] == 1]
    df_not_clicked = df[df['clicked'] == 0]
    min_class_size = min(len(df_clicked), len(df_not_clicked))
    n = min(min_class_size, 2_500_000) 
    df_sampled = pd.concat([
        df_clicked.sample(n=n, random_state=42),
        df_not_clicked.sample(n=n, random_state=42)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)
    df = df_sampled
    print(df['clicked'].value_counts(normalize=True))

    if df['new_session'].dtype == 'bool':
        df['new_session'] = df['new_session'].astype(int)

    for col in ['publish_time_admeta', 'publish_time_eventmeta']:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        df[col + '_ts'] = df[col].astype('int64') // 10**9 
        df[col + '_year'] = df[col].dt.year
        df[col + '_month'] = df[col].dt.month

    cat_cols = [
        'display_id', 'ad_id', 'ad_document_id', 'campaign_id', 'advertiser_id', 'uuid',
        'event_document_id', 'platform_x', 'geo_location_x', 'source_id_admeta', 'publisher_id_x',
        'source_id_eventmeta', 'publisher_id_y', 'document_id_x', 'category_id', 'document_id_ent',
        'entity_id', 'document_id_y', 'platform_y', 'geo_location_y', 'traffic_source', 'session_id'
    ]

    for col in cat_cols:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str).fillna('NA'))
        df[col] = df[col].fillna(-1).astype(int)

    dense_cols = ['confidence_level', 'confidence_level_ent', 'timestamp', 'time_diff']
    for col in dense_cols:
        df[col] = df[col].fillna(0).astype(float)

    drop_cols = []
    for col in ['hour', 'dayofweek', 'session_id']:
        if df[col].nunique() == 1:
            drop_cols.append(col)
    df = df.drop(columns=drop_cols)

    print("Done encoding. Final columns:")
    print(df.dtypes)

    if 'new_session' in df.columns and df['new_session'].dtype == 'bool':
        df['new_session'] = df['new_session'].astype(int)
    date_cols = []
    for col in df.columns:
        if (
            'publish_time' in col or 'timestamp' in col
        ) and (
            df[col].dtype == 'object' or pd.api.types.is_datetime64_any_dtype(df[col])
        ):
            date_cols.append(col)
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        df[col + '_ts'] = df[col].astype('int64') // 10**9
        df[col + '_year'] = df[col].dt.year
        df[col + '_month'] = df[col].dt.month
    df = df.drop(columns=date_cols)
    for col in df.select_dtypes(include=['object', 'category']):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str).fillna('NA'))
    for col in df.select_dtypes(include=['float', 'int']):
        if df[col].isnull().any():
            fill_value = -1 if pd.api.types.is_integer_dtype(df[col]) else 0
            df[col] = df[col].fillna(fill_value)

    constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) == 1]
    df = df.drop(columns=constant_cols)

    print("Final DataFrame shape:", df.shape)
    print("Columns used for modeling:", df.columns.tolist())

    target_col = 'clicked'

    train_val_df, test_df = train_test_split(
        df, 
        test_size=0.15, 
        random_state=42, 
        shuffle=True,
        stratify=df[target_col]
    )

    val_frac_of_train_val = 0.15 / (1 - 0.15) 

    train_df, valid_df = train_test_split(
        train_val_df,
        test_size=val_frac_of_train_val,
        random_state=42,
        shuffle=True,
        stratify=train_val_df[target_col]
    )
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    valid_df.to_csv(os.path.join(output_dir, 'valid.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    print("Files saved: train.csv, valid.csv, test.csv")
    print("Train:",train_df['clicked'].value_counts(normalize=True))
    print()
    print("Validation:",valid_df['clicked'].value_counts(normalize=True))
    print()
    print("Test:",test_df['clicked'].value_counts(normalize=True))
