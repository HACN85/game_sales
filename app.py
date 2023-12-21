import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Anchor element for the top of the page
st.markdown('<div id="top"></div>', unsafe_allow_html=True)

st.set_option('deprecation.showPyplotGlobalUse', False)


# Function to plot distribution of Global Sales
def plot_global_sales_distribution(data):
    st.write("### Distribution of Global Sales")
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Global_Sales'], kde=True)
    plt.xlabel('Global Sales')
    plt.ylabel('Frequency')
    st.pyplot()

# Load the CSV file
@st.cache_data
def load_data():
    file_path = "vgsales.csv"
    data = pd.read_csv(file_path)
    return data

def filter_data(data, platforms, years, genres, publishers):
    # Function to filter data based on user selections
    filtered_data = data.copy()

    # Create a mapping of indices for each unique value
    unique_values = {
        'Platform': data['Platform'].unique(),
        'Year': sorted(data['Year'].unique()),
        'Genre': data['Genre'].unique(),
        'Publisher': data['Publisher'].unique()
    }

    # Sort the selected filter values based on their original order
    for category, selected_values in [('Platform', platforms), ('Year', years), ('Genre', genres), ('Publisher', publishers)]:
        if selected_values and 'Select All' not in selected_values:
            sorted_values = sorted(selected_values, key=lambda x: unique_values[category].tolist().index(x))
            filtered_data = filtered_data[filtered_data[category].isin(sorted_values)]

    return filtered_data



# Function to plot sales trend over years
def plot_sales_over_years(data):
    st.write("### Sales Trend Over Years")
    sales_by_year = data.groupby('Year')['Global_Sales'].sum()
    plt.figure(figsize=(10, 6))
    sales_by_year.plot(kind='line')
    plt.xlabel('Year')
    plt.ylabel('Global Sales')
    st.pyplot()

# Function to compare sales across regions
def plot_sales_across_regions(data):
    st.write("### Sales Comparison Across Regions")
    sales_regions = data[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]
    sales_regions_sum = sales_regions.sum()
    plt.figure(figsize=(10, 6))
    sales_regions_sum.plot(kind='bar')
    plt.xlabel('Region')
    plt.ylabel('Total Sales')
    st.pyplot()

# Function to show market share by Publisher
def plot_market_share_publisher(data):
    st.write("### Market Share by Publisher")
    publisher_sales = data.groupby('Publisher')['Global_Sales'].sum().sort_values(ascending=False)[:10]
    plt.figure(figsize=(10, 6))
    publisher_sales.plot(kind='pie', autopct='%1.1f%%')
    plt.ylabel('')
    st.pyplot()

# Function to show market share by Platform
def plot_market_share_platform(data):
    st.write("### Market Share by Platform")
    platform_sales = data.groupby('Platform')['Global_Sales'].sum().sort_values(ascending=False)[:10]
    plt.figure(figsize=(10, 6))
    platform_sales.plot(kind='pie', autopct='%1.1f%%')
    plt.ylabel('')
    st.pyplot()

# Function to display boxplots for Sales by Genre
def plot_sales_by_genre(data):
    st.write("### Sales Distribution by Genre")
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Genre', y='Global_Sales', data=data)
    plt.xticks(rotation=90)
    plt.xlabel('Genre')
    plt.ylabel('Global Sales')
    st.pyplot()


# Function to show correlation heatmap
def plot_correlation_heatmap(data):
    st.write("### Correlation Heatmap")
    st.write("The correlation heatmap visualizes the relationships between numerical variables in the dataset.")
    st.write(
        "Each cell in the heatmap represents the correlation coefficient between two variables. The values range from -1 to 1.")
    st.write(
        "A value closer to 1 implies a strong positive correlation, while a value closer to -1 implies a strong negative correlation.")
    st.write("A value around 0 suggests little to no linear relationship between the variables.")

    numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
    numerical_data = data[numerical_cols]

    plt.figure(figsize=(10, 8))
    sns.heatmap(numerical_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    st.pyplot()


# Function to show scatter plot matrix
def plot_scatter_plot_matrix(data):
    st.write("### Scatter Plot Matrix")
    st.write(
        "The scatter plot matrix provides a grid of scatterplots showcasing pairwise relationships between numerical variables.")
    st.write("Each cell in the matrix represents the relationship between two variables.")
    st.write("The diagonal line of plots displays the distribution of each variable.")

    fig = px.scatter_matrix(data, dimensions=data.select_dtypes(include=['float64', 'int64']).columns)
    fig.update_traces(diagonal_visible=False)  # Hides the distribution plot on the diagonal
    fig.update_layout(width=800, height=800)
    st.plotly_chart(fig)

def main():
    st.title('Game Sales Data Explorer')
    st.write("""
    This dataset is a window into the top 100-selling video games, presenting key details like platforms, genres, and publishers. Analyzing this Kaggle dataset reveals platform popularity, successful genres on specific platforms, and standout publishers.

    To leverage this dataset effectively, start by exploring each column's significance. With 11 columns, including rank, title, platform, release year, genre, publisher, and sales figures, comparisons and industry trends become apparent over time.

    Use visualizations like graphs and charts for comprehensive insights, revealing nuances not immediately evident in raw data. Refine searches with keywords for precise data extraction and detailed analysis!
    """)

    data = load_data()
    st.write('## Game Sales Data')
    st.write(data.head())  # Display the first few rows of the dataset

    # Filter options with "Select All" option
    platforms = st.multiselect('Select Platforms', ['Select All'] + list(data['Platform'].unique()))
    years = st.multiselect('Select Years', ['Select All'] + [str(year) for year in data['Year'].unique()])
    genres = st.multiselect('Select Genres', ['Select All'] + list(data['Genre'].unique()))
    publishers = st.multiselect('Select Publishers', ['Select All'] + list(data['Publisher'].unique()))

    # Apply filters
    filtered_data = filter_data(data, platforms, years, genres, publishers)

    # Display filtered data
    st.write('## Filtered Data')
    st.write(filtered_data)

    # Numerical columns for regression
    numerical_columns = [col for col in filtered_data.columns if filtered_data[col].dtype in ['int64', 'float64']]
    numerical_factors = st.multiselect('Select Numerical Factors for Regression', numerical_columns)

    if numerical_factors:
        st.write("### Numerical Factors for Regression")
        st.write(
            "The selected numerical factors for regression analysis display the relationship between each factor and the "
            "game's Global Sales. Regression analysis helps us understand how changes in these factors relate to changes in "
            "the game's sales performance. The plots below showcase these relationships."
        )
        for factor in numerical_factors:
            if factor != 'Global_Sales':  # Avoid plotting Global_Sales against itself
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.regplot(x=factor, y='Global_Sales', data=filtered_data, ax=ax)
                ax.set_title(f'{factor} vs Global Sales')
                ax.set_xlabel(factor)
                ax.set_ylabel('Global Sales')
                st.pyplot(fig)
                st.write("---")

    # Call additional visualization functions
    if st.checkbox('Show Additional Insights'):
        # Call the plotting functions here...
        plot_global_sales_distribution(filtered_data)
        plot_sales_over_years(filtered_data)
        plot_sales_across_regions(filtered_data)
        plot_market_share_publisher(filtered_data)
        plot_market_share_platform(filtered_data)
        plot_sales_by_genre(filtered_data)
        plot_correlation_heatmap(filtered_data)
        plot_scatter_plot_matrix(filtered_data)

        # Button to scroll back to the top
        st.markdown('<p style="text-align:center;"><a href="#top" onclick="window.scrollTo(0, 0);">Back to Top</a></p>',
                    unsafe_allow_html=True)

if __name__ == "__main__":
    main()
