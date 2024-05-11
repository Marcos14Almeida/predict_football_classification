import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def plot(data):

    data = data.loc[:, (data < 2).any()]

    # Plotting
    plt.figure(figsize=(10, 6))
    for club in data.columns:
        plt.plot(data.index, data[club], marker='o', label=club)

    plt.xlabel('Season')
    plt.ylabel('Position')
    plt.title('Club Positions Over Seasons')
    plt.legend()
    plt.ylim(0, 20)
    plt.xticks(data.index)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Assuming you have the historical data stored in a CSV file named 'football_data.csv'
data = pd.read_csv('assets/europa_top.csv')

# Assuming the CSV file has columns: 'League', 'Year', 'Pos', 'Team'
# Filter the data for the Brasileirão league and the years
min_year = 2005
prediciton_year = 2023
league = 'La Liga'
league2 = ''

league_dataset = data[(data['League'] == league) | (data['League'] == league2)]
league_dataset.loc[league_dataset['League'] == league2, 'Pos'] += 20
data_filtered = league_dataset[(league_dataset['Year'] >= min_year) & (league_dataset['Year'] < prediciton_year)]

if len(league2) > 3:
    get_names = data_filtered[
        ((data['League'] == league) & (data['Year'] == prediciton_year - 1) & (data['Pos'] <= 16)) 
        | ((data['League'] == league2) & (data['Year'] == prediciton_year - 1) & (data['Pos'] <= 4))
        ]
else:
    get_names = data_filtered[
        ((data['League'] == league) & (data['Year'] == prediciton_year - 1) & (data['Pos'] <= 16))
        ]

names_1stdivision = get_names['Team'].unique()
data_filtered = data_filtered[data_filtered['Team'].isin(names_1stdivision)]
print(data_filtered)


# Pivot the DataFrame
pivot_df = data_filtered.pivot(index='Year', columns='Team', values='Pos')

pivot_df = pivot_df.fillna(50)

# Display the pivoted DataFrame
print(pivot_df)

pivot_df.to_csv('assets/filtered.csv', index=True)

df = pivot_df

# plot(df)

# Create an empty DataFrame to store the predicted values
predicted_df = pd.DataFrame(index=df.index, columns=df.columns)

predictions = {}
# Iterate over each column in the DataFrame
for column in df.columns:

    # Get the values for the current column
    X = df[column].shift().dropna().values.reshape(-1, 1)  # Use previous values as features
    y = df[column].dropna().values[1:]  # Target variable is the next value, remove the first value

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict the next value
    next_value = model.predict(X[-1].reshape(1, -1))
    predictions[column] = next_value[0]

# Extract column names and their corresponding values
columns = list(predictions.keys())
values = [value for value in predictions.values()]

# Sort the column names based on their values
sorted_columns = [column for _, column in sorted(zip(values, columns))]

sorted_df = pd.DataFrame({'Delete': sorted_columns})
sorted_df['Pos'] = sorted_df.index + 1
sorted_df.drop(columns=['Delete'], inplace=True)
sorted_df['Team'] = sorted_columns

# Error
real_df = league_dataset[(league_dataset['Year'] == prediciton_year)]
if not real_df.empty:
    sum = 0
    diff_map = {}
    for index, predict_row in sorted_df.iterrows():
        real_row = real_df[real_df['Team'] == predict_row['Team']]
        try:
            diff = predict_row['Pos'] - real_row['Pos'].iloc[0]
        except:
            diff = 0
        diff_map[predict_row['Team']] = diff
        sum += abs(diff)

    diff_df = pd.DataFrame(diff_map.items(), columns=['Team_predicted', 'Pos'])

    real_df.drop(columns=['League', 'Year'], inplace=True)
    final_df = pd.merge(real_df, sorted_df, on='Pos', how='left')
    final_df.rename(columns={'Team_x': 'Team_real'}, inplace=True)
    final_df.rename(columns={'Team_y': 'Team_predicted'}, inplace=True)

    final_df = pd.merge(final_df, diff_df, on='Team_predicted', how='left')
    final_df.rename(columns={'Pos_x': 'Pos'}, inplace=True)
    final_df.rename(columns={'Pos_y': 'Difference'}, inplace=True)
    final_df = final_df[['Pos', 'Team_predicted', 'Difference', 'Team_real']]

    print(f"\nPREDICTION {prediciton_year}:")
    print(final_df)

    print("\nAverage Error:", sum / len(diff_map))

else:
    print(f"\nPREDICTION {prediciton_year}:")
    for index, predict_row in sorted_df.iterrows():
        pos = predict_row["Pos"]
        team = predict_row["Team"]
        print(f"{pos}º {team}")
