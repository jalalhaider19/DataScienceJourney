import pandas as pd 

# reading the data set 
df = pd.read_csv("drowsiness_dataset.csv")

# See the first few lines 
print(df.head())

# Dataset Summary 
print(df.describe())

# Checking for any null values 
print(df.isnull().sum())

# Dividing data in the 4 periods 
num_periods = 4 
period_size = len(df) // num_periods 

# The data set is divided in the 4 time periods 
period_labels = ["Morning","Afternoon","Evening","Night"]
df['period'] = pd.cut(df.index, bins = num_periods, labels= period_labels)

print(df['period'].value_counts())

morning_data = df[df['period'] == 'Morning']
afternoon_data = df[df['period'] == 'Afternoon']
evening_data = df[df['period'] == 'Evening']
night_data = df[df['period'] == 'Night']


import matplotlib.pyplot as plt 
import seaborn as sns

# Calculate & Plot correlations b/w Data & periods 
def calculate_and_plot_correlations(data, period_name):
    correlation_heart_rate = data['drowsiness'].corr(data['heartRate'])
    correlation_heart_ppg_green = data['drowsiness'].corr(data['ppgGreen'])
    correlation_heart_ppg_red = data['drowsiness'].corr(data['ppgRed'])
    correlation_heart_ppg_ir = data['drowsiness'].corr(data['ppgIR'])

    print(f'Correlation between drowsiness and heart rate({period_name}): {correlation_heart_rate}')
    print(f'Correlation between drowsiness and PPG green({period_name}): {correlation_heart_ppg_green}')
    print(f'Correlation between drowsiness and PPG red({period_name}): {correlation_heart_ppg_red}')
    print(f'Correlation between drowsiness and PPG Infrared({period_name}): {correlation_heart_ppg_ir}')

    plt.scatter(data['heartRate'], data['drowsiness'], alpha = 0.5, label='Heart Rate')
    plt.scatter(data['ppgGreen'], data['drowsiness'], alpha = 0.5, label='PPG Green', color = 'green')
    plt.scatter(data['ppgRed'], data['drowsiness'], alpha = 0.5, label='PPG Red', color = 'red')
    plt.scatter(data['ppgIR'], data['drowsiness'], alpha = 0.5, label='PPG Purple', color = 'purple')
    plt.title(f'Drowsiness Levels vs Heart Rate & PPG Readings({period_name})')
    plt.xlabel('Activity Level')
    plt.ylabel('Drowsiness Level')
    plt.legend()
    plt.show()


calculate_and_plot_correlations(morning_data, 'Morning')
calculate_and_plot_correlations(afternoon_data, 'Afternoon')
calculate_and_plot_correlations(evening_data, 'Evening')
calculate_and_plot_correlations(night_data, 'Night')


#Histogram for numerical columns
df.hist(bins = 20, figsize=(12, 9))
plt.show()
