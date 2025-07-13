#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


clicks_test = pd.read_csv('clicks_test.csv')
clicks_train = pd.read_csv('clicks_train.csv')
documents_categories = pd.read_csv('documents_categories.csv')
documents_entities = pd.read_csv('documents_entities.csv')
documents_meta = pd.read_csv('documents_meta.csv')
documents_topics = pd.read_csv('documents_topics.csv')
events = pd.read_csv('events.csv')
page_views_sample = pd.read_csv('page_views_sample.csv')
promoted_content = pd.read_csv('promoted_content.csv')


# In[3]:


#Taking top 5000000 rows, we dont need to use it
clicks_train=clicks_train.iloc[:5000000,:]


# In[4]:


#Taking top 5000000 rows, we dont need to use it
unique_ad_ids = clicks_train['ad_id'].unique().tolist()


# In[5]:


#Taking top 5000000 rows, we dont need to use it
promoted_content = promoted_content[promoted_content['ad_id'].isin(unique_ad_ids)]


# In[6]:


print("Step 1: Merging clicks_train with promoted_content...")
df = clicks_train.merge(promoted_content, how='left', on='ad_id')
df.rename(columns={'document_id': 'ad_document_id'}, inplace=True)


# In[7]:


#Taking top 5000000 rows, we dont need to use it
unique_display_ids = events['display_id'].unique().tolist()


# In[8]:


#Taking top 5000000 rows, we dont need to use it
events = events[events['display_id'].isin(unique_display_ids)]


# In[9]:


# Step 2: Merge with events
print("Step 2: Merging with events...")
df = df.merge(events, how='left', on='display_id')
df.rename(columns={'document_id': 'event_document_id', 'timestamp': 'event_timestamp'}, inplace=True)


# In[10]:


#Taking top 5000000 rows, we dont need to use it
unique_ad_document_ids = df['ad_document_id'].unique().tolist()


# In[11]:


#Taking top 5000000 rows, we dont need to use it
documents_meta = documents_meta[documents_meta['document_id'].isin(unique_ad_document_ids)]


# In[12]:


# Step 3a: Merge with metadata for ad_document_id
print("Step 3a: Merging ad_document_id with documents_meta...")
df = df.merge(documents_meta, how='left', left_on='ad_document_id', right_on='document_id')
df.rename(columns=lambda x: f"{x}_admeta" if x in ['source_id', 'published_id', 'publish_time'] else x, inplace=True)
df.drop(columns=['document_id'], inplace=True)


# In[13]:


#Taking top 5000000 rows, we dont need to use it
unique_event_document_id = df['event_document_id'].unique().tolist()


# In[14]:


#Taking top 5000000 rows, we dont need to use it
documents_meta  = documents_meta[documents_meta['document_id'].isin(unique_event_document_id)]


# In[15]:


# Step 3b: Merge with metadata for event_document_id
print("Step 3b: Merging event_document_id with documents_meta...")
df = df.merge(documents_meta, how='left', left_on='event_document_id', right_on='document_id')
df.rename(columns=lambda x: f"{x}_eventmeta" if x in ['source_id', 'published_id', 'publish_time'] else x, inplace=True)
df.drop(columns=['document_id'], inplace=True)


# In[16]:


#Taking top 5000000 rows, we dont need to use it
unique_ad_document_id = df['ad_document_id'].unique().tolist()


# In[17]:


#Taking top 5000000 rows, we dont need to use it
documents_categories  = documents_categories[documents_categories['document_id'].isin(unique_ad_document_id)]


# In[18]:


# Step 4: Merge with topics, categories, and entities
print("Step 4: Merging with documents_categories...")
df = df.merge(documents_categories, how='left', left_on='ad_document_id', right_on='document_id', suffixes=('', '_cat'))


# In[19]:


#Taking top 5000000 rows, we dont need to use it
unique_ad_document_id = df['ad_document_id'].unique().tolist()


# In[20]:


#Taking top 5000000 rows, we dont need to use it
documents_entities  = documents_entities[documents_entities['document_id'].isin(unique_ad_document_id)]


# In[21]:


print("Step 4: Merging with documents_entities...")
df = df.merge(documents_entities, how='left', left_on='ad_document_id', right_on='document_id', suffixes=('', '_ent'))


# In[22]:


#Taking top 5000000 rows, we dont need to use it
unique_uuid = df['uuid'].unique().tolist()


# In[23]:


#Taking top 5000000 rows, we dont need to use it
unique_event_document_id= df['event_document_id'].unique().tolist()


# In[24]:


#Taking top 5000000 rows, we dont need to use it
page_views_sample  = page_views_sample[page_views_sample['uuid'].isin(unique_uuid)]
page_views_sample  = page_views_sample[page_views_sample['document_id'].isin(unique_event_document_id)]


# In[25]:


# Step 5: Merge with page_views_sample
print("Step 5: Merging with page_views_sample...")
df = df.merge(page_views_sample, how='left', left_on=['uuid', 'event_document_id'], right_on=['uuid', 'document_id'])


# In[26]:


# Step 6: Extract temporal features
print("Step 6: Extracting temporal features...")
df['event_timestamp'] = pd.to_datetime(df['event_timestamp'], unit='ms')
df['hour'] = df['event_timestamp'].dt.hour
df['dayofweek'] = df['event_timestamp'].dt.dayofweek


# In[27]:


df = df.sort_values(by=['uuid', 'event_timestamp']).reset_index(drop=True)

# Compute the time difference (in minutes) for each user's events
df['time_diff'] = df.groupby('uuid')['event_timestamp'].diff().dt.total_seconds().div(60)

# Start a new session if the time difference is more than 30 minutes or it's the first row for that uuid
df['new_session'] = (df['time_diff'] > 30) | (df['time_diff'].isna())

# Session id: cumulative sum of new sessions per user
df['session_id'] = df.groupby('uuid')['new_session'].cumsum()

print("? All steps completed. Final shape:", df.shape)


# In[28]:


df


# In[29]:


df = df.iloc[:5000000,:]


# In[30]:


# Separate classes
df_clicked = df[df['clicked'] == 1]
df_not_clicked = df[df['clicked'] == 0]

# Get min count from both classes to sample equally
min_class_size = min(len(df_clicked), len(df_not_clicked))
n = min(min_class_size, 2_500_000)  # Cap at 2.5M if both classes allow

# Sample equally from both classes
df_sampled = pd.concat([
    df_clicked.sample(n=n, random_state=42),
    df_not_clicked.sample(n=n, random_state=42)
]).sample(frac=1, random_state=42).reset_index(drop=True)

# Overwrite original
df = df_sampled

# Verify class balance
print(df['clicked'].value_counts(normalize=True))


# In[31]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load your data
#df = pd.read_excel('final_data_3.xlsx')

# Boolean to int
if df['new_session'].dtype == 'bool':
    df['new_session'] = df['new_session'].astype(int)

# Convert string/object dates to datetime, then to timestamp (float)
for col in ['publish_time_admeta', 'publish_time_eventmeta']:
    df[col] = pd.to_datetime(df[col], errors='coerce')
    # Option 1: Replace column with timestamp (seconds since epoch)
    df[col + '_ts'] = df[col].astype('int64') // 10**9  # create new col for timestamp
    # Option 2: Extract year/month/day if you want those as features
    df[col + '_year'] = df[col].dt.year
    df[col + '_month'] = df[col].dt.month

# For categorical columns with possible -1 for missing, ensure int (they are already)
cat_cols = [
    'display_id', 'ad_id', 'ad_document_id', 'campaign_id', 'advertiser_id', 'uuid',
    'event_document_id', 'platform_x', 'geo_location_x', 'source_id_admeta', 'publisher_id_x',
    'source_id_eventmeta', 'publisher_id_y', 'document_id_x', 'category_id', 'document_id_ent',
    'entity_id', 'document_id_y', 'platform_y', 'geo_location_y', 'traffic_source', 'session_id'
]

# If you want to label encode any that are truly categorical (not just IDs)
for col in cat_cols:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str).fillna('NA'))
    # Fill missing if you want (if -1 is not missing already)
    df[col] = df[col].fillna(-1).astype(int)

# Float/dense columns
dense_cols = ['confidence_level', 'confidence_level_ent', 'timestamp', 'time_diff']
for col in dense_cols:
    df[col] = df[col].fillna(0).astype(float)

# Drop columns with only one value (won't help ML)
drop_cols = []
for col in ['hour', 'dayofweek', 'session_id']:
    if df[col].nunique() == 1:
        drop_cols.append(col)
df = df.drop(columns=drop_cols)

# If you want, print your feature matrix info
print("Done encoding. Final columns:")
print(df.dtypes)


# In[32]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder

#df = pd.read_excel('final_data_4.xlsx')

# 1. Boolean to int
if 'new_session' in df.columns and df['new_session'].dtype == 'bool':
    df['new_session'] = df['new_session'].astype(int)

# 2. Identify date columns (object or datetime) for encoding
date_cols = []
for col in df.columns:
    if (
        'publish_time' in col or 'timestamp' in col
    ) and (
        df[col].dtype == 'object' or pd.api.types.is_datetime64_any_dtype(df[col])
    ):
        date_cols.append(col)

# 3. Encode datetime columns (and drop originals)
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')
    df[col + '_ts'] = df[col].astype('int64') // 10**9
    df[col + '_year'] = df[col].dt.year
    df[col + '_month'] = df[col].dt.month
df = df.drop(columns=date_cols)

# 4. Encode string/categorical columns
for col in df.select_dtypes(include=['object', 'category']):
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str).fillna('NA'))

# 5. Fill missing values for numeric columns
for col in df.select_dtypes(include=['float', 'int']):
    if df[col].isnull().any():
        fill_value = -1 if pd.api.types.is_integer_dtype(df[col]) else 0
        df[col] = df[col].fillna(fill_value)

# 6. Drop constant columns
constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) == 1]
df = df.drop(columns=constant_cols)

print("Final DataFrame shape:", df.shape)
print("Columns used for modeling:", df.columns.tolist())


# In[33]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Assume 'clicked' is your target column
target_col = 'clicked'

# First, split off the test set (15%) with stratification
train_val_df, test_df = train_test_split(
    df, 
    test_size=0.15, 
    random_state=42, 
    shuffle=True,
    stratify=df[target_col]
)

# Next, split train+val into train and validation (val = 15% of original)
val_frac_of_train_val = 0.15 / (1 - 0.15)  # ? 0.1765

train_df, valid_df = train_test_split(
    train_val_df,
    test_size=val_frac_of_train_val,
    random_state=42,
    shuffle=True,
    stratify=train_val_df[target_col]
)

# Save the splits
train_df.to_csv('train.csv', index=False)
valid_df.to_csv('valid.csv', index=False)
test_df.to_csv('test.csv', index=False)

print("Files saved: train.csv, valid.csv, test.csv")


# In[34]:


# Verify class balance in train, test and validation sets
print("Train:",train_df['clicked'].value_counts(normalize=True))
print()
print("Validation:",valid_df['clicked'].value_counts(normalize=True))
print()
print("Test:",test_df['clicked'].value_counts(normalize=True))

