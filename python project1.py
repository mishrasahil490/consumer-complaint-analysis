import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('C:/Users/mishr/OneDrive/Desktop/Python Project/Consumer Complaints.csv')

# -------------------Objective 1: Clean the dataset----------------------------------------------------------------------------------------------
def clean_data(df):
    df['Sub-issue'] = df['Sub-issue'].replace('', 'Not Specified').fillna('Not Specified')
    df['State'] = df['State'].fillna('Unknown')
    df['Timely response?'] = df['Timely response?'].fillna('Unknown')
    df['Date submitted'] = pd.to_datetime(df['Date submitted'], errors='coerce')
    df['Date received'] = pd.to_datetime(df['Date received'], errors='coerce')
    df = df.drop_duplicates()
    return df

data = clean_data(data)
print("Objective 1: Data cleaned. Missing values handled, dates converted.")
print("Sample of cleaned data:")
print(data.head())

#-----------------Objective 2: Visualize distribution of complaints by state------------------------------------------------------------------
def plot_complaints_by_state(df):
    plt.figure(figsize=(12, 6))
    state_counts = df['State'].value_counts().head(10)
    sns.barplot(x=state_counts.values, y=state_counts.index, hue=state_counts.index, palette='muted', legend=False)
    plt.title('Top 10 States by Complaint Count')
    plt.xlabel('Number of Complaints')
    plt.ylabel('State')
    plt.tight_layout()
    plt.show()
    
    top_5_states = df['State'].value_counts().head(5)
    others = df['State'].value_counts()[5:].sum()
    pie_data = pd.concat([top_5_states, pd.Series({'Others': others})])
    plt.figure(figsize=(8, 8))
    plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'), startangle=90)
    plt.title('Proportion of Complaints: Top 5 States vs. Others')
    plt.show()
    
    print("\nState Complaint Statistics:")
    print(df['State'].value_counts().describe())

plot_complaints_by_state(data)
print("Objective 2: Visualized and analyzed complaints by state.")

#--------------Objective 3: Analyze the frequency of complaint issues--------------------------------------------------------------------------
def analyze_complaint_issues(df):
    issue_counts = df['Issue'].value_counts().head(10)
    print("\nTop 10 Complaint Issues:")
    print(issue_counts)
    
    # Bar plot with top issue highlighted
    plt.figure(figsize=(10, 6))
    colors = ['red' if i == 0 else 'skyblue' for i in range(len(issue_counts))]  # Highlight top issue
    sns.barplot(y=issue_counts.index, x=issue_counts.values, hue=issue_counts.index, palette=colors, legend=False)
    plt.title('Top 10 Complaint Issues (Top Issue Highlighted)')
    plt.xlabel('Number of Complaints')
    plt.ylabel('Issue')
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Issue'].value_counts(), bins=30, kde=True, color='teal')
    plt.title('Distribution of Complaint Counts Across Issues')
    plt.xlabel('Number of Complaints per Issue')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
    
    total_complaints = len(df)
    top_issue_percent = (issue_counts.iloc[0] / total_complaints) * 100
    print(f"Most common issue accounts for {top_issue_percent:.2f}% of complaints.")

analyze_complaint_issues(data)
print("Objective 3: Analyzed frequency of complaint issues.")

#---------------Objective 4: Create a complaint severity index-----------------------------------------------------------------------------------
def complaint_severity_index(df):
    df['Severity Score'] = df['Company response to consumer'].map({
        'Closed with monetary relief': 3,
        'Closed with non-monetary relief': 3,
        'Closed with explanation': 2,
        'In progress': 1,
        'Closed': 1
    }).fillna(1)
    df['Severity Score'] = df.apply(
        lambda x: x['Severity Score'] - 1 if x['Timely response?'] == 'No' else x['Severity Score'], axis=1)
    
    severity_by_product = df.groupby('Product')['Severity Score'].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=severity_by_product.values, y=severity_by_product.index, hue=severity_by_product.index, 
                palette='rocket', legend=False)
    plt.title('Complaint Severity Index by Product')
    plt.xlabel('Average Severity Score')
    plt.ylabel('Product')
    plt.tight_layout()
    plt.show()
    
    pivot_table = df.pivot_table(values='Severity Score', index='Product', 
                                columns='Timely response?', aggfunc='mean', fill_value=0)
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, cmap='YlOrRd', fmt='.2f', 
                cbar_kws={'label': 'Severity Score', 'ticks': [0, 1, 2, 3]})
    plt.title('Severity Score vs. Timely Response by Product')
    plt.xlabel('Timely Response')
    plt.ylabel('Product')
    plt.tight_layout()
    plt.show()
    
    print("\nSeverity Score Statistics:")
    print(severity_by_product.describe())
    
    top_severity = severity_by_product.idxmax()
    print(f"\nCreative Insight: {top_severity} has the highest severity index, "
          "indicating urgent need for proactive resolution strategies.")

complaint_severity_index(data)
print("Objective 4: Created and analyzed complaint severity index.")

#--------------------Objective 5: Explore complaint trends over time--------------------------------------------------------------
def complaint_trends_over_time(df):
    df['YearMonth'] = df['Date submitted'].dt.to_period('M')
    time_trends = df.groupby(['YearMonth', 'Product']).size().unstack(fill_value=0)
    top_products = df['Product'].value_counts().head(3).index
    time_trends = time_trends[top_products]
    
    plt.figure(figsize=(12, 6))
    time_trends.plot(kind='line', marker='o', colormap='tab10')
    plt.title('Complaint Trends Over Time for Top 3 Products')
    plt.xlabel('Year-Month')
    plt.ylabel('Number of Complaints')
    plt.xticks(rotation=45)
    plt.legend(title='Product')
    plt.tight_layout()
    plt.show()
    
    print("\nMonthly Complaint Counts (Top Product):")
    top_product = df['Product'].value_counts().idxmax()
    monthly_counts = df[df['Product'] == top_product].groupby('YearMonth').size()
    print(monthly_counts.describe())
    
    peak_month = monthly_counts.idxmax()
    print(f"\nCreative Insight: Complaints for {top_product} peaked in {peak_month}, "
          "suggesting seasonal campaigns or system issues to investigate.")

complaint_trends_over_time(data)
print("Objective 5: Explored complaint trends over time.")

#----------------------Objective 6: Design a customer impact score---------------------------------------------------------------
def customer_impact_score(df):
    issue_freq = df['Issue'].value_counts() / len(df)
    resolution_weight = df['Company response to consumer'].map({
        'Closed with monetary relief': 0.5,
        'Closed with non-monetary relief': 0.5,
        'Closed with explanation': 0.3,
        'In progress': 0.2,
        'Closed': 0.1
    }).fillna(0.1)
    df['Resolution Weight'] = resolution_weight
    
    impact_by_issue = df.groupby('Issue')['Resolution Weight'].mean() * issue_freq
    impact_by_issue = impact_by_issue.sort_values(ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=impact_by_issue.values, y=impact_by_issue.index, size=impact_by_issue.values, 
                    hue=impact_by_issue.values, palette='flare', legend=False)
    mean_impact = impact_by_issue.mean()
    plt.axvline(x=mean_impact, color='gray', linestyle='--', label='Mean Impact')
    for i, score in enumerate(impact_by_issue.values):
        plt.text(score, i, f'{score:.4f}', ha='left', va='center', fontsize=10)
    plt.title('Customer Impact Score by Issue (Top 10)')
    plt.xlabel('Impact Score')
    plt.ylabel('Issue')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    print("\nCustomer Impact Score Statistics:")
    print(impact_by_issue.describe())
    
    top_issue = impact_by_issue.idxmax()
    print(f"\nCreative Insight: '{top_issue}' has the highest customer impact score, "
          "recommending targeted consumer education or process automation.")

customer_impact_score(data)
print("Objective 6: Designed and analyzed customer impact score.")

#-----------------Additional EDA and Statistical Analysis---------------------------------------------------------------------
def additional_eda(df):
    print("\nSummary Statistics for Key Columns:")
    for col in ['Product', 'Timely response?', 'Company response to consumer']:
        print(f"\n{col}:")
        print(df[col].value_counts())
    
    df['Response Time'] = (df['Date received'] - df['Date submitted']).dt.days
    print("\nResponse Time (Days) Statistics:")
    print(df['Response Time'].describe())
    
    df['YearMonth'] = df['Date submitted'].dt.to_period('M')
    response_time_trend = df.groupby('YearMonth')['Response Time'].mean().dropna()
    plt.figure(figsize=(12, 6))
    response_time_trend.plot(kind='line', marker='s', color='purple')
    plt.title('Average Response Time Trend Over Time')
    plt.xlabel('Year-Month')
    plt.ylabel('Average Response Time (Days)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    response_times = df['Response Time'].dropna()
    if not response_times.empty:
        Q1 = response_times.quantile(0.25)
        Q3 = response_times.quantile(0.75)
        IQR = Q3 - Q1
        outliers = response_times[(response_times < (Q1 - 1.5 * IQR)) | (response_times > (Q3 + 1.5 * IQR))]
        print(f"\nNumber of Response Time Outliers: {len(outliers)}")
    else:
        print("\nNo response time data available for outlier detection.")
    
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Response Time'].dropna(), bins=20, kde=True)
    plt.title('Distribution of Response Time (Days)')
    plt.xlabel('Response Time (Days)')
    plt.ylabel('Frequency')
    plt.show()

additional_eda(data)
print("Performed additional EDA and statistical analysis.")
